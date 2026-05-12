import torch
from src.models.submodules.backbones import HRNetBackbone
from src.models.submodules.necks import HRNetNeck
import torch.nn.functional as F


class CrowdCounter(torch.nn.Module):
    def __init__(self, params,):
        super().__init__()
        
        # 1. Initialize Backbone
        self.backbone = HRNetBackbone(params)
        
        # 2. Automatically determine the channel sizes for the neck
        # We pass a dummy tensor through the backbone to see what channel sizes it outputs.
        # This makes your code robust; if you change hrnet_w18 to hrnet_w48, you don't break the code!
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            dummy_features = self.backbone(dummy_input)
        in_channels_list = [f.shape[1] for f in dummy_features]
        
        # 3. Initialize Neck
        self.neck = HRNetNeck(in_channels_list=in_channels_list, out_channels=params.neck_out_channels)
        
        # 4. Final Counting Head (Outputs a 1-Channel Density Map)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(params.neck_out_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 1, kernel_size=1) # 1 channel output for density estimation
        )

    def forward(self, x):
        input_size = x.shape[2:] # Save the original image size (H, W)
        
        # Extract features
        features = self.backbone(x)
        
        # Fuse features in the neck
        fused = self.neck(features)
        
        # Predict the density map (currently at 1/4 resolution due to HRNet's f1)
        density_map = self.head(fused)
        
        # Upsample the predicted density map back to the original image resolution
        density_map = F.interpolate(density_map, size=input_size, mode='bilinear', align_corners=False)
        
        # Ensure no negative values in the density map
        density_map = F.relu(density_map)
        
        return density_map