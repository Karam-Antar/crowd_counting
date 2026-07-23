import torch
from src import config
import torch.nn.functional as F

from src.data.transform import UnpadToOriginal

@torch.no_grad()
def predict(model, x: torch.Tensor, device=config.device, h=None, w=None):
        """
        Custom inference method for single or batched inputs.
        """
        # 1. Handle dimensionality (C, H, W) -> (1, C, H, W)
        if not isinstance(model, torch.fx.GraphModule):
            model.eval()
        x = x.to(device)

        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # 2. Forward pass
        density_map = model(x)
        
        # 3. Squeeze the channel dimension IMMEDIATELY
        # Converts [Batch, 1, Height, Width] -> [Batch, Height, Width]
        if density_map.dim() == 4 and density_map.shape[1] == 1:
            density_map = density_map.squeeze(1)
        
        # 4. Ensure no negative predictions
        # density_map = F.softplus(density_map) 
        
        # 5. The total count is the sum of the density map.
        # Note: Because we removed the channel dimension, density_map is now 3D [B, H, W].
        # We sum over H (dim 1) and W (dim 2).
        if h and w:
            density_map = UnpadToOriginal()(density_map, original_shape=(h, w))
        total_counts = density_map.sum(dim=(1, 2))
        
        return total_counts, density_map