import torch
from src import config
import torch.nn.functional as F

from src.data.transform import UnpadToOriginal

@torch.no_grad()
def predict(model, x: torch.Tensor, device=config.device, h=None, w=None):
    if not isinstance(model, torch.fx.GraphModule):
        model.eval()
    
    x = x.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)
                
    # 1. WRAP THE FORWARD PASS IN AUTOCAST
    # This forces intermediate tensors to use 16-bit precision, halving memory
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    if device_type == 'cuda':
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            density_map = model(x)
    else:
        density_map = model(x)
            
    if density_map.dim() == 4 and density_map.shape[1] == 1:
        density_map = density_map.squeeze(1)
            
    # Cast back to float32 for final math operations to ensure stability
    density_map = density_map.float()
    
    if h and w:
        density_map = UnpadToOriginal()(density_map, original_shape=(h, w))
    total_counts = density_map.sum(dim=(1, 2))
            
    return total_counts, density_map
