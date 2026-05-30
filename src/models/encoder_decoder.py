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
        # Note: SMP automatically builds the backbone, the U-Net skip connections, 
        # and the final regression head.
        smp_class = getattr(smp, params.model_class)  # You can easily switch to 'FPN', 'DeepLabV3', etc. by changing this string
        self.net = smp_class(
            encoder_name=params.backbone,       # e.g., 'resnet34', 'efficientnet-b3', or 'tu-hrnet_w18'
            encoder_weights=params.backbone_weights,
            in_channels=3,
            classes=1,                          # Output a 1-channel density map
        )
        
        # 2. Handle Backbone Freezing safely
        # In SMP, the backbone is perfectly isolated inside `self.model.encoder`
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
        # 1. Forward Pass
        # SMP's U-Net naturally processes the skip connections and inherently 
        # upsamples the output all the way back to the exact (H, W) of the input.
        density_map = self.net(x)
        
        return density_map