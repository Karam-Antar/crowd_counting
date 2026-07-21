import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from src.core.params import BaseParams

class EncoderDecoder(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.params = params
        
        # 1. Initialize SMP U-Net
        # The output of this will now act as a feature extractor.
        # Ensure params.neck_out_channels > 1 (e.g., 16 or 32) in your config.
        smp_class = getattr(smp, params.model_class)  
        self.net = smp_class(
            encoder_name=params.backbone,       
            encoder_weights=params.backbone_weights,
            decoder_attention_type=params.decoder_attention_type,           
            in_channels=3,
            classes=params.neck_out_channels, 
        )

        # --- NEW: The Dual-Branch Gating Mechanism ---
        # Branch A: Reduces the feature map down to a 1-channel raw density prediction
        self.density_head = nn.Sequential(
            nn.Conv2d(params.neck_out_channels, 1, kernel_size=1),
            nn.Softplus() # Ensures the network cannot predict negative people
        )
        
        # Branch B: Evaluates the same features to generate a 0-to-1 background mask
        self.attention_head = nn.Sequential(
            nn.Conv2d(params.neck_out_channels, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        # ---------------------------------------------
        
        # 2. Handle Backbone Freezing safely
        if not params.trainable_backbone:
            for param in self.net.encoder.parameters():
                param.requires_grad = False
                
        elif params.unfrozen_blocks:
            for param in self.net.encoder.parameters():
                param.requires_grad = False
                
            for name, param in self.net.encoder.named_parameters():
                if any(block_name in name for block_name in params.unfrozen_blocks):
                    param.requires_grad = True

    def forward(self, x):
        # 1. Extract rich semantic features from SMP
        features = self.net(x)
        
        # 2. Generate raw density and the attention mask in parallel
        raw_density = self.density_head(features)
        spatial_mask = self.attention_head(features)
        
        # 3. Apply the gate
        # This instantly zero-outs raw density values where the mask predicts background
        final_density = raw_density * spatial_mask
        
        # Returning only the final tensor prevents breaking your existing training loop
        return final_density