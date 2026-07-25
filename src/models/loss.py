import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure
import segmentation_models_pytorch as smp
from src.core.params import BaseParams

class MSESSIMLoss(nn.Module):
    def __init__(self, params: BaseParams):
        """
        Args:
            ssim_weight (float): Balances pixel intensity (MSE) vs structure (SSIM).
                                 0.1 to 0.2 is typically ideal for crowd counting.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = params.ssim_weight
        self.mse_weight = 1.0 - self.ssim_weight

    def forward(self, pred_density, gt_density):
        # 1. Calculate MSE (This directly optimizes your PSNR safely)
        loss_mse = self.mse(pred_density, gt_density)
        
        # 2. Calculate SSIM Loss
        max_val = torch.clamp(gt_density.max(), min=1e-5)
        
        ssim_score = structural_similarity_index_measure(
            pred_density, gt_density, data_range=max_val
        )
        loss_ssim = 1.0 - ssim_score
        
        # 3. Combine them
        total_loss = (self.mse_weight * loss_mse) + (self.ssim_weight * loss_ssim)
        return total_loss


class SSIMLoss(nn.Module):
    def __init__(self):
        """
        Pure SSIM Loss module for density map structural optimization.
        Optimizes strictly for spatial layout, contrast, and structural patterns.
        """
        super().__init__()

    def forward(self, pred_density, gt_density):
        # 1. Dynamically determine data range based on the ground truth batch max.
        max_val = max(gt_density.max().item(), 1e-5)
        
        # 2. Calculate the structural similarity index measure score
        # (torchmetrics outputs 1.0 for a perfect structural match)
        ssim_score = structural_similarity_index_measure(
            pred_density, gt_density, data_range=max_val
        )
        
        # 3. Return 1 - SSIM so that a perfect match minimizes down to 0.0
        loss_ssim = 1.0 - ssim_score
        
        return loss_ssim


class CountPenaltyLoss(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        self.ssim_weight = params.ssim_weight
        self.mse_weight = 1.0 - self.ssim_weight
        
        # Keep this small. We want to guide the network away from background 
        # false positives, not overpower the pixel-level layout.
        self.count_weight = 0.08 

    def forward(self, pred_density, gt_density):
        # 1. Pixel-level Loss (No extra scaling needed since labels are x1000)
        loss_mse = self.mse(pred_density, gt_density)
        
        # 2. Structural Loss
        # Safely find the max value in the current batch on the GPU without .item()
        # Adding 1e-5 prevents division by zero on entirely empty background patches
        batch_max = torch.max(gt_density) + 1e-5
        
        ssim_score = structural_similarity_index_measure(
            pred_density, gt_density, data_range=batch_max
        )
        loss_ssim = 1.0 - ssim_score
        
        # 3. Global Count Loss
        pred_count = pred_density.sum(dim=[1, 2, 3])
        gt_count = gt_density.sum(dim=[1, 2, 3])
        
        # Divide by 1000 so the L1 loss represents the true person-count error,
        # otherwise this loss term will be artificially inflated by 1000x.
        loss_count = self.l1(pred_count, gt_count)
        
        # 4. Combine
        total_loss = (self.mse_weight * loss_mse) + \
                     (self.ssim_weight * loss_ssim) + \
                     (self.count_weight * loss_count)
                     
        return total_loss

class MaskMSESSIMLoss(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.mse = nn.HuberLoss(delta=params.huber_delta)
        self.mask_loss_fn = smp.losses.FocalLoss(
            mode='binary',
            alpha=0.75,
            gamma=2.0
        )
        # self.bg_penalty_weight = bg_penalty_weight
        
        # 4 learnable parameters for [MSE, SSIM, Mask, Background Penalty]
        # Kendall et al. uncertainty parameters (initialized to 0 -> exp(0) = 1)
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.label_scaler = params.label_scaler

    def forward(self, pred_density, mask_logits, gt_density):
        # 1. Density Loss (Huber)
        raw_mse = self.mse(pred_density, gt_density)
        
        # 2. SSIM Loss with FIXED data_range to prevent gradient explosion on zero-density images
        # 0.5 is standard for normalized density map peaks; keep fixed across all batches
        ssim_score = structural_similarity_index_measure(pred_density, gt_density, data_range=1*self.label_scaler)
        raw_ssim = 1.0 - ssim_score
        
        # 3. Mask Loss with Thresholded Gaussian Tails
        gt_mask = (gt_density > 5e-4*self.label_scaler).float()  # Cut off Gaussian tails
        
        if mask_logits is not None:
            raw_mask = self.mask_loss_fn(mask_logits, gt_mask)
        else:
            raw_mask = torch.tensor(0.0, device=pred_density.device)


        # --- Kendall Uncertainty Weighting Formulation ---
        # Formula: 0.5 * exp(-log_var) * Loss + 0.5 * log_var
        loss_mse = 0.5 * torch.exp(-self.log_vars[0]) * raw_mse + 0.5 * self.log_vars[0]
        loss_ssim = 0.5 * torch.exp(-self.log_vars[1]) * raw_ssim + 0.5 * self.log_vars[1]
        loss_mask = 0.5 * torch.exp(-self.log_vars[2]) * raw_mask + 0.5 * self.log_vars[2]

        total_loss = loss_mse + loss_ssim + loss_mask
            
        return total_loss


# class MaskMSESSIMLoss(nn.Module):
#     def __init__(self, params: BaseParams):
#         super().__init__()
#         self.mse = nn.HuberLoss(delta=params.huber_delta)
#         self.mask_loss_fn = smp.losses.FocalLoss(
#             mode='binary',
#             alpha=0.75, # Weight for the positive class (foreground)
#             gamma=2.0   # Focusing parameter (2.0 is standard)
#         )
#         self.ssim_weight = params.ssim_weight
#         self.mse_weight = 1-self.ssim_weight
#         self.mask_loss_weight = params.mask_loss_weight
#         # 3 learnable parameters for MSE, SSIM, and Mask
#         # Initialized to 0 (since they represent log(variance))
#         self.log_vars = nn.Parameter(torch.zeros(3))

#     def forward(self, pred_density, mask_logits, gt_density):
#         raw_mse = self.mse(pred_density, gt_density) * self.mse_weight
        
#         max_val = torch.clamp(gt_density.max(), min=1e-5)
#         ssim_score = structural_similarity_index_measure(pred_density, gt_density, data_range=max_val) * self.ssim_weight
#         raw_ssim = 1.0 - ssim_score
        
#         # Formula: (Loss / (2 * exp(log_var))) + (log_var / 2)
#         # The log_var term penalizes the network for just making the denominator huge
#         loss_mse = (raw_mse * torch.exp(-self.log_vars[0])) + self.log_vars[0]
#         loss_ssim = (raw_ssim * torch.exp(-self.log_vars[1])) + self.log_vars[1]
        
#         total_loss = loss_mse + loss_ssim
        
#         if mask_logits is not None:
#             gt_mask = (gt_density > 0).float()
#             raw_mask = self.mask_loss_fn(mask_logits, gt_mask)
#             loss_mask = (raw_mask * torch.exp(-self.log_vars[2])) + self.log_vars[2]
#             total_loss += self.mask_loss_weight * loss_mask
#             # print(f"MSE: {raw_mse.item():.4f} | SSIM: {raw_ssim.item():.4f} | Mask: {raw_mask.item():.4f}")
#             # print(f"Scaled MSE: {loss_mse.item():.4f} | Scaled SSIM: {loss_ssim.item():.4f} | Scaled Mask: {loss_mask.item():.4f}")
#             # print()
            
#         return total_loss



class SpatiallyWeightedLoss(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        # reduction='none' allows us to weight pixels individually
        self.mse = nn.MSELoss(reduction='none')
        self.ssim_weight = params.ssim_weight
        self.mse_weight = 1.0 - self.ssim_weight
        self.alpha = 3.0   # Background penalty multiplier
        self.gamma = 0.1   # Decay factor for density

    def forward(self, pred_density, gt_density):
        # 1. Pixel-wise MSE
        raw_mse = self.mse(pred_density, gt_density)
        
        # 2. Continuous weight map: Empty areas get higher weight (alpha), 
        # dense crowd areas naturally decay down to a weight of 1.0
        weight_map = 1.0 + (self.alpha - 1.0) * torch.exp(-self.gamma * gt_density)
        
        loss_mse = torch.mean(raw_mse * weight_map)
        
        # 3. SSIM Loss
        max_val = torch.clamp(gt_density.max(), min=1e-5)
        loss_ssim = 1.0 - structural_similarity_index_measure(pred_density, gt_density, data_range=max_val)
        
        return (self.mse_weight * loss_mse) + (self.ssim_weight * loss_ssim)