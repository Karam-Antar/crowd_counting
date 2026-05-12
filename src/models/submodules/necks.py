import torch
import torch.nn as nn
import torch.nn.functional as F

class HRNetNeck(nn.Module):
    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        # HRNet extracts features at different scales, but with different channel depths.
        # We sum up the channels because we will concatenate them.
        total_in_channels = sum(in_channels_list)
        
        # A simple fusion block to compress the concatenated features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features is a list: [f1, f2, f3, f4]
        # f1 has the highest resolution (1/4th of the original image)
        target_size = features[0].shape[2:] 

        upsampled_features = [features[0]]
        
        # Iterate over the deeper/smaller feature maps and upsample them to match f1
        for i in range(1, len(features)):
            upsampled = F.interpolate(features[i], size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(upsampled)

        # Concatenate all features along the channel dimension
        # As discussed in your documents: torch.cat([f1, f2, f3, f4], dim=1)
        out = torch.cat(upsampled_features, dim=1)
        
        # Apply the fusion convolution
        out = self.fusion_conv(out)
        return out