import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure

class HybridMSESSIMLoss(nn.Module):
    def __init__(self, ssim_weight=0.2):
        """
        Args:
            ssim_weight (float): Balances pixel intensity (MSE) vs structure (SSIM).
                                 0.1 to 0.2 is typically ideal for crowd counting.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = ssim_weight

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
        total_loss = loss_mse + (self.ssim_weight * loss_ssim)
        return total_loss