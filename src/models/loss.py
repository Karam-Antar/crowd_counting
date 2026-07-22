import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure

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


class BCEMSESSIMLoss(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss() # NEW: To train the mask
        self.ssim_weight = params.ssim_weight
        self.mse_weight = 1.0 - self.ssim_weight

    def forward(self, pred_density, spatial_mask, gt_density):
        # 1. Standard Density Losses
        loss_mse = self.mse(pred_density, gt_density)
        
        max_val = torch.clamp(gt_density.max(), min=1e-5)
        ssim_score = structural_similarity_index_measure(
            pred_density, gt_density, data_range=max_val
        )
        loss_ssim = 1.0 - ssim_score
        
        # 2. Explicit Mask Supervision
        # Create a binary mask: 1.0 where there is any density, 0.0 for pure background.
        # (Assuming your GT is scaled by 1000, 1e-4 is a safe threshold)
        gt_mask = (gt_density > 0).float() 
        loss_mask = self.bce(spatial_mask, gt_mask)
        
        # 3. Combine
        # The mask loss usually needs a small weight (e.g., 0.1) so it doesn't overpower MSE
        total_loss = (self.mse_weight * loss_mse) + \
                     (self.ssim_weight * loss_ssim) + \
                     (0.1 * loss_mask)
                     
        return total_loss