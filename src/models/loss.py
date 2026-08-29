import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure
import segmentation_models_pytorch as smp
from src.core.params import BaseParams

class MSESSIMLoss(nn.Module):
    """Combine pixel-wise MSE and structural SSIM into a composite density-map loss."""
    def __init__(self, params: BaseParams):
        """Initialize the MSE+SSIM composite loss.

        Args:
            params (BaseParams): Parameter bundle containing the SSIM weight.
        """
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
        """Compute the MSE-SSIM composite loss for a prediction-target density pair.

        Args:
            pred_density (torch.Tensor): Predicted density map batch.
            gt_density (torch.Tensor): Ground-truth density map batch.

        Returns:
            torch.Tensor: Scalar composite loss value.
        """
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
    """Loss that optimizes density-map structure without an explicit MSE term."""
    def __init__(self):
        """Initialize the structure-only SSIM loss module."""
        """
        Pure SSIM Loss module for density map structural optimization.
        Optimizes strictly for spatial layout, contrast, and structural patterns.
        """
        super().__init__()

    def forward(self, pred_density, gt_density):
        """Compute the complement of SSIM between predicted and ground-truth density maps.

        Args:
            pred_density (torch.Tensor): Predicted density map batch.
            gt_density (torch.Tensor): Ground-truth density map batch.

        Returns:
            torch.Tensor: Scalar loss value where zero indicates a perfect structural match.
        """
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
    """Add a global count regularizer on top of pixel-level density-map losses."""
    def __init__(self, params: BaseParams):
        """Initialize the count-penalty objective.

        Args:
            params (BaseParams): Parameter bundle containing the SSIM mix weight.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        self.ssim_weight = params.ssim_weight
        self.mse_weight = 1.0 - self.ssim_weight
        
        # Keep this small. We want to guide the network away from background 
        # false positives, not overpower the pixel-level layout.
        self.count_weight = 0.08 

    def forward(self, pred_density, gt_density):
        """Compute MSE, SSIM, and count-penalty losses for the current batch.

        Args:
            pred_density (torch.Tensor): Predicted density-map batch.
            gt_density (torch.Tensor): Ground-truth density-map batch.

        Returns:
            torch.Tensor: Scalar combined loss.
        """
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
    """Combine density regression, SSIM, and foreground-mask supervision in one module."""
    def __init__(self, params: BaseParams):
        """Initialize the mask-aware composite loss.

        Args:
            params (BaseParams): Parameter bundle controlling Huber loss, mask focal loss,
                and uncertainty weighting.
        """
        super().__init__()
        self.mse = nn.HuberLoss(delta=params.huber_delta)
        self.mask_loss_fn = smp.losses.FocalLoss(
            mode='binary',
            alpha=params.mask_loss_alpha, # Weight for the positive class (foreground)
            gamma=params.mask_loss_gamma,   # Focusing parameter (2.0 is standard)
        )
        self.ssim_weight = params.ssim_weight
        self.mse_weight = 1-self.ssim_weight
        self.mask_loss_weight = params.mask_loss_weight
        # self.use_uncertainty = params.use_uncertainty_weighting
        # 3 learnable parameters for MSE, SSIM, and Mask
        # Initialized to 0 (since they represent log(variance))
        self.log_vars = nn.Parameter(torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)) if params.use_uncertainty_weighting else None
        self.gt_mask_threshold = params.gt_mask_threshold

    def forward(self, pred_density, mask_logits, gt_density):
        """Compute the full mask-aware composite loss and associated sub-loss breakdown.

        Args:
            pred_density (torch.Tensor): Predicted density map batch.
            mask_logits (torch.Tensor): Foreground logit mask before sigmoid activation.
            gt_density (torch.Tensor): Ground-truth density target batch.

        Returns:
            tuple[torch.Tensor, dict]: Combined scalar loss and dictionary of component losses.
        """
        raw_mse = self.mse(pred_density, gt_density) * self.mse_weight
        
        max_val = torch.clamp(gt_density.max(), min=1e-5)
        ssim_score = structural_similarity_index_measure(pred_density, gt_density, data_range=max_val) * self.ssim_weight
        raw_ssim = 1.0 - ssim_score
        
        # Formula: (Loss / (2 * exp(log_var))) + (log_var / 2)
        # The log_var term penalizes the network for just making the denominator huge
        if self.log_vars:
            loss_mse = (raw_mse * torch.exp(-self.log_vars[0])) + self.log_vars[0]
            loss_ssim = (raw_ssim * torch.exp(-self.log_vars[1])) + self.log_vars[1]
        else:
            loss_mse = raw_mse
            loss_ssim = raw_ssim

        total_loss = loss_mse + loss_ssim
        # Prepare loss dictionary to track individual parts
        loss_dict = {
            'loss_mse': loss_mse,
            'loss_ssim': loss_ssim,
            'raw_mse': raw_mse,
            'raw_ssim': raw_ssim,
        }
        if self.log_vars:
            loss_dict['log_var_mse'] = self.log_vars[0]
            loss_dict['log_var_ssim'] = self.log_vars[1]
        
        if mask_logits is not None:
            gt_mask = (gt_density > self.gt_mask_threshold).float()
            raw_mask = self.mask_loss_fn(mask_logits, gt_mask)
            if self.log_vars:
                loss_mask = (raw_mask * torch.exp(-self.log_vars[2])) + self.log_vars[2]
            else:
                loss_mask = raw_mask
            total_loss += self.mask_loss_weight * loss_mask
            # print(f"MSE: {raw_mse.item():.4f} | SSIM: {raw_ssim.item():.4f} | Mask: {raw_mask.item():.4f}")
            # print(f"Scaled MSE: {loss_mse.item():.4f} | Scaled SSIM: {loss_ssim.item():.4f} | Scaled Mask: {loss_mask.item():.4f}")
            # print()
            loss_dict['loss_mask'] = loss_mask
            loss_dict['raw_mask'] = raw_mask
            if self.log_vars:
                loss_dict['log_var_mask'] = self.log_vars[2]
            
        return total_loss, loss_dict



class SpatiallyWeightedLoss(nn.Module):
    """Use a foreground-aware weighting map to emphasize crowded and sparse regions."""
    def __init__(self, params: BaseParams):
        """Initialize the spatially weighted loss.

        Args:
            params (BaseParams): Parameter bundle controlling the SSIM mix weight.
        """
        super().__init__()
        # reduction='none' allows us to weight pixels individually
        self.mse = nn.MSELoss(reduction='none')
        self.ssim_weight = params.ssim_weight
        self.mse_weight = 1.0 - self.ssim_weight
        self.alpha = 3.0   # Background penalty multiplier
        self.gamma = 0.1   # Decay factor for density

    def forward(self, pred_density, gt_density):
        """Compute a density-map loss weighted by the local density distribution.

        Args:
            pred_density (torch.Tensor): Predicted density map batch.
            gt_density (torch.Tensor): Ground-truth density map batch.

        Returns:
            torch.Tensor: Scalar combined spatially weighted loss.
        """
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