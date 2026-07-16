import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure

from src.core.params import BaseParams

class HybridMSESSIMLoss(nn.Module):
    def __init__(self, params: BaseParams):
        """
        Args:
            ssim_weight (float): Balances pixel intensity (MSE) vs structure (SSIM).
                                 0.1 to 0.2 is typically ideal for crowd counting.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = params.ssim_weight
        self.mse_weight = 1-self.ssim_weight

    def forward(self, pred_density, gt_density):
        # 1. Calculate MSE (This directly optimizes your PSNR safely)
        loss_mse = self.mse(pred_density, gt_density)
        
        # 2. Calculate SSIM Loss
        max_val = max(gt_density.max().item(), 1e-5)
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