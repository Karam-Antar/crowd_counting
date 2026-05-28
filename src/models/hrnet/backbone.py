import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HRNetBackbone(nn.Module):
    def __init__(self, params, model_name='hrnet_w18', pretrained=True):
        super().__init__()
        # Using features_only=True tells timm to return intermediate feature maps
        # For HRNet, this returns a list of 4 feature maps at different resolutions
        # (usually 1/4, 1/8, 1/16, 1/32 of the input image size)
        self.backbone = timm.create_model(params.backbone, pretrained=pretrained, features_only=True)
        if not params.trainable_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif params.unfrozen_blocks:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Note: Parameter names in timm vary by model, 
            # but usually follow standard naming conventions.
            for name, param in self.backbone.named_parameters():
                if any(block_name in name for block_name in params.unfrozen_blocks):
                    param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)