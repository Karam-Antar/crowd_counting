import torch
from torch import nn
from src import config
from src.core.params import BaseParams
from src.models.hrnet.net import HRNet
import torch.nn.functional as F

from src.models.encoder_decoder import EncoderDecoder


class CrowdCounter(torch.nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.params = params
        # 1. Initialize Backbone
        if params.backbone.startswith('hrnet'):
            self.net = HRNet(params)
        else:
            self.net = EncoderDecoder(params)
        
        if self.params.loss_function == 'bce_mse_ssim':
            # --- NEW: The Dual-Branch Gating Mechanism ---
            # Branch A: Reduces the feature map down to a 1-channel raw density prediction
            self.density_head = nn.Sequential(
                nn.Conv2d(params.decoder_out_channels, 1, kernel_size=1),
                nn.Softplus() # Ensures the network cannot predict negative people
            )
            
            # Branch B: Evaluates the same features to generate a 0-to-1 background mask
            self.attention_head = nn.Sequential(
                nn.Conv2d(params.decoder_out_channels, 1, kernel_size=1),
                nn.Sigmoid() 
            )
            # ---------------------------------------------
        

    def forward(self, x):
        features = self.net(x)
        final_density = features
        if self.params.loss_function == 'bce_mse_ssim':
            # 2. Generate raw density and the attention mask in parallel
            raw_density = self.density_head(features)
            spatial_mask = self.attention_head(features)
            
            # 3. Apply the gate
            # This instantly zero-outs raw density values where the mask predicts background
            final_density = raw_density * spatial_mask
            if self.training:
                return final_density, spatial_mask
            else:
                scaler_val = float(self.params.label_scaler) 
                final_density = final_density / scaler_val
                return final_density
        # Ensure no negative values in the density map
        # final_density = F.softplus(final_density)
        if not self.training:
            scaler_val = float(self.params.label_scaler) 
            final_density = final_density / scaler_val
        
        return final_density
    

    def sliding_window_inference(self, images: torch.Tensor, device=config.device) -> torch.Tensor:
        """
        Reusable method that handles padding, patching, and recombining 
        the image using a sliding window. 
        """
        window_size = self.params.crop_size
        stride: int = window_size // 2
        B, C, H, W = images.shape
        
        # 1. Pad if necessary
        pad_h = max(0, window_size - H)
        pad_w = max(0, window_size - W)
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))
            
        _, _, curr_h, curr_w = images.shape
        
        # 2. Prepare accumulators
        full_density_map = torch.zeros((B, curr_h, curr_w), device=device)
        overlap_count = torch.zeros((B, curr_h, curr_w), device=device)
        
        y_coords = list(range(0, curr_h - window_size + 1, stride))
        x_coords = list(range(0, curr_w - window_size + 1, stride))
        
        if (curr_h > window_size) and (y_coords[-1] + window_size < curr_h):
            y_coords.append(curr_h - window_size)
        if (curr_w > window_size) and (x_coords[-1] + window_size < curr_w):
            x_coords.append(curr_w - window_size)
            
        # 3. Patch and Accumulate
        for y in y_coords:
            for x_c in x_coords:
                patch = images[:, :, y:y+window_size, x_c:x_c+window_size]
                
                # Normal forward pass on the patch
                patch_density = self(patch) 
                
                if patch_density.dim() == 4 and patch_density.shape[1] == 1:
                    patch_density = patch_density.squeeze(1)
                    
                full_density_map[:, y:y+window_size, x_c:x_c+window_size] += patch_density
                overlap_count[:, y:y+window_size, x_c:x_c+window_size] += 1
        
        # 4. Average and Crop
        full_density_map = full_density_map / (overlap_count + 1e-6)
        
        if pad_h > 0 or pad_w > 0:
            full_density_map = full_density_map[:, :H, :W]
            
        return full_density_map