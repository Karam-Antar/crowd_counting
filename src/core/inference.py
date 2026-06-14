import torch
from src import config
import torch.nn.functional as F

@torch.no_grad()
def predict(model, x: torch.Tensor, device=config.device):
        """
        Custom inference method for single inputs.
        """
        """
        Custom inference method for single inputs.
        """
        # 1. Handle dimensionality (C, H, W) -> (1, C, H, W)
        if not isinstance(model, torch.fx.GraphModule):
            model.eval()
        x = x.to(device)

        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        density_map = model(x) / config.LABEL_SCALER
        
        # Ensure no negative predictions
        density_map = F.relu(density_map) 
        
        # The total count is the integral (sum) of the density map
        total_counts = density_map.sum(dim=(1, 2, 3))
        
        return total_counts, density_map