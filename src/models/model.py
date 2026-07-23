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
        self.threshold = self.params.seg_threshold
        # 1. Initialize Backbone
        if params.backbone.startswith('hrnet'):
            self.net = HRNet(params)
        else:
            self.net = EncoderDecoder(params)
        
        if self.params.loss_function == 'mask_mse_ssim':
           # A good rule of thumb is to halve or keep the channel count of the backbone's output
            mid_channels = params.decoder_out_channels // 2 
            dropout_rate = self.params.dropout or 0.0
            # Branch A: Density (Regression)
            self.density_head = nn.Sequential(
                # 1. Private spatial processing for regression
                nn.Conv2d(params.decoder_out_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                # 2. Final linear projection
                nn.Conv2d(mid_channels, 1, kernel_size=1),
                nn.ReLU() # Ensures no negative density
            )

            # Branch B: Attention Mask (Binary Classification)
            self.attention_head = nn.Sequential(
                # 1. Private spatial processing for edge/boundary detection
                nn.Conv2d(params.decoder_out_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                # 2. Final linear projection
                nn.Conv2d(mid_channels, 1, kernel_size=1)
            )

            # Initialize the final bias of the attention head (index 2 now, because of the mid-layer)
            nn.init.constant_(self.attention_head[3].bias, 1.2)
            # ---------------------------------------------
        

    def forward(self, x, return_mask=False):
        features = self.net(x)
        
        if self.params.loss_function != 'mask_mse_ssim':
            final_density = features
            if not self.training:
                final_density = final_density / float(self.params.label_scaler)
            return final_density

        # 1. Raw outputs (CRITICAL: apply ReLU to density so it can't be negative)
        raw_density = self.density_head(features)
        mask_logits = self.attention_head(features) 
        
        # 2. Gate the density with the mask for the final output
        spatial_mask = torch.sigmoid(mask_logits)
        # Detach isn't strictly necessary since we evaluate loss on the raw outputs, 
        # but we can leave it to be safe.
        spatial_mask_binary = (spatial_mask > self.threshold).float()

        # 3. Multiply by the binary mask
        final_density = raw_density.detach() * spatial_mask_binary.detach()
        
        # 3. Handle Scaling specifically for Evaluation/Inference
        if not self.training:
            scaler_val = float(self.params.label_scaler) 
            # Scale down the gated density for accurate metric counting
            final_density = final_density / scaler_val
            
            # NOTE: If your validation dataset provides 'y' that is already scaled DOWN, 
            # you also need to divide raw_density by scaler_val here so the validation loss is correct.
            # raw_density = raw_density / scaler_val

        # 4. Return routing
        if self.training or return_mask:
            return final_density, raw_density, mask_logits
        
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


# mid_channels = params.decoder_out_channels // 2 
# dropout_rate = self.params.dropout # Keep it low (20%) to regularize without starving the network

# # Branch A: Density (Regression)
# self.density_head = nn.Sequential(
#     # 1. Dilation increases receptive field for scale variation
#     nn.Conv2d(params.decoder_out_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
#     nn.BatchNorm2d(mid_channels),    # Stabilizes gradients
#     nn.ReLU(inplace=True),
#     nn.Dropout2d(p=dropout_rate),    # Anti-overfitting
#     # 2. Final linear projection
#     nn.Conv2d(mid_channels, 1, kernel_size=1),
#     nn.ReLU() 
# )

# # Branch B: Attention Mask (Binary Classification)
# self.attention_head = nn.Sequential(
#     # 1. Standard convolution for sharp boundary detection
#     nn.Conv2d(params.decoder_out_channels, mid_channels, kernel_size=3, padding=1),
#     nn.BatchNorm2d(mid_channels),    # Stabilizes gradients
#     nn.ReLU(inplace=True),
#     nn.Dropout2d(p=dropout_rate),    # Anti-overfitting
#     # 2. Final linear projection
#     nn.Conv2d(mid_channels, 1, kernel_size=1)
# )

# # Initialize the final bias of the attention head 
# # (Index is now 4 because of the added BN and Dropout layers: Conv(0)->BN(1)->ReLU(2)->Drop(3)->Conv(4))
# nn.init.constant_(self.attention_head[4].bias, 1.3)