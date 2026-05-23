import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    def __init__(self, in_channels, out_channels=None, reduction=32):
        super(CoordAtt, self).__init__()
        
        # 1. Safety default: Attention usually preserves the channel depth.
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 2. Projection Layer: If in_channels differs from out_channels, 
        # we MUST project the identity map, otherwise the final multiplication will crash.
        if self.in_channels != self.out_channels:
            self.project_identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.project_identity = nn.Identity()

        # Calculate a reduced intermediate channel size to save parameters
        mip = max(8, in_channels // reduction)

        # 3. Shared transformations
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        # 4. Separate transformations to create attention maps for X and Y
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Project identity if channels differ, otherwise it does nothing
        identity = self.project_identity(x)
        
        n, c, h, w = x.size()

        # Extract X and Y coordinate vectors 
        # (Using torch.mean is mathematically identical to AdaptiveAvgPool2d but safer for compilers)
        x_h = torch.mean(x, dim=3, keepdim=True)  # Shape: [N, C, H, 1]
        x_w = torch.mean(x, dim=2, keepdim=True)  # Shape: [N, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)             # Shape: [N, C, W, 1]

        # Concatenate and apply shared convolution
        y = torch.cat([x_h, x_w], dim=2)          # Shape: [N, C, H + W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into X and Y 
        # (Using tensor slicing instead of torch.split prevents dynamic shape crashes)
        x_h = y[:, :, :h, :]                      # Shape: [N, mip, H, 1]
        x_w = y[:, :, h:, :]                      # Shape: [N, mip, W, 1]
        
        x_w = x_w.permute(0, 1, 3, 2)             # Flip X back to [N, mip, 1, W]

        # Generate Attention Weights (0 to 1)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        # Apply the "crosshairs" attention back to the feature map
        out = identity * a_w * a_h

        return out