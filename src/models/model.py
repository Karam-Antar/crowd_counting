import torch
from torch import nn
from src import config
from src.core.params import BaseParams
import torch.nn.functional as F

from src.models.encoder_decoder import EncoderDecoder


class CrowdCounter(torch.nn.Module):
    """Density-estimation network that maps RGB images to crowd density maps.

    The model uses an encoder-decoder backbone for feature extraction, optionally adds a
    foreground-attention mask head for segmentation-aware training, and returns either a
    density map or a pair of ``(density_map, mask_logits)`` depending on the selected loss.

    Attributes:
        params (BaseParams): Config controlling the architecture and scale behavior.
        net (EncoderDecoder): Feature extraction backbone.
        density_features: Optional convolutional head producing density features.
        final_density_conv: Optional final density map projection.
        attention_head: Optional foreground attention head used for mask-aware losses.
    """
    def __init__(self, params: BaseParams):
        """Initialize the crowd counter architecture.

        Args:
            params (BaseParams): Parameter bundle describing the selected backbone and loss
                configuration.
        """
        super().__init__()
        self.params = params
        # 1. Initialize Backbone
        self.net = EncoderDecoder(params)
        
        if self.params.loss_function == 'mask_mse_ssim':
            # A good rule of thumb is to halve or keep the channel count of the backbone's output
            mid_channels = self.params.decoder_out_channels // 2 
            dropout_rate = self.params.dropout or 0.0
            
            # --- STAGE 1: DENSITY FEATURE EXTRACTOR ---
            self.density_features = nn.Sequential(
                nn.Conv2d(self.params.decoder_out_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate)
            )
            
            # Final linear projection for density
            self.final_density_conv = nn.Sequential(
                nn.Conv2d(mid_channels, 1, kernel_size=1),
                nn.ReLU() # Ensures no negative density
            )

            # --- STAGE 2: ATTENTION MASK (Macro Spatial Filter) ---
            self.attention_head = nn.Sequential(
                nn.Conv2d(self.params.decoder_out_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(mid_channels, 1, kernel_size=1)
            )
            # Initialize the final bias of the attention head
            nn.init.constant_(self.attention_head[-1].bias, 1.2)
            
        

    def forward(self, x, return_mask=False):
        """Run the crowd density model on an input image batch.

        Args:
            x (torch.Tensor): Batch of normalized RGB images with shape ``(B, 3, H, W)``.
            return_mask (bool): If ``True``, return the mask logits alongside the density
                map when the mask-aware loss is active.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Density map output, and
                optionally foreground mask logits for the mask-aware loss.
        """
        features = self.net(x)
        
        if self.params.loss_function != 'mask_mse_ssim':
            final_density = features
            if not self.training:
                final_density = final_density / float(self.params.label_scaler)
            return final_density

        # --- APPLY MACRO SPATIAL GATING ---
        mask_logits = self.attention_head(features) 
        spatial_mask = torch.sigmoid(mask_logits)
        
        den_feats = self.density_features(features)
        
        # Soft gate the density features with a residual connection
        gated_feats = (den_feats * spatial_mask) + den_feats
        
        # Generate the raw density map (will still have microscopic noise)
        raw_density = self.final_density_conv(gated_feats)

        
        # 2. Scale the threshold up to match your 1000x density scale
        scaler_val = float(self.params.label_scaler)

        final_density = raw_density
        # --- Handle Scaling for Evaluation/Inference ---
        if not self.training:
            # Scale down the gated density for accurate metric counting
            final_density = final_density / scaler_val

        # --- Return Routing ---
        if self.training or return_mask:
            return final_density, mask_logits
        
        return final_density