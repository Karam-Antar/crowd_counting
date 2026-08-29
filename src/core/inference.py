import torch
from src import config
import torch.nn.functional as F

from src.data.transform import UnpadToOriginal

@torch.no_grad()
def predict(model, x: torch.Tensor, device=config.device, h=None, w=None):
    """Run single or batched crowd-count inference and return counts plus density maps.

    Args:
        model: A trained PyTorch model that accepts image tensors shaped like
            ``(B, C, H, W)`` and returns either a density map tensor or a tuple of
            density map plus auxiliary outputs.
        x (torch.Tensor): Input image tensor. Batches are accepted as ``(B, C, H, W)``;
            single images may be passed as ``(C, H, W)`` and are expanded to a batch.
        device: Target device used for inference, such as ``"cuda"`` or ``"cpu"``.
        h (Optional[int]): Original image height before padding. Used to crop the
            model output back to the original spatial size.
        w (Optional[int]): Original image width before padding. Used with ``h`` to
            unpad the model output.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple ``(total_counts, density_map)`` where
            ``total_counts`` has shape ``(B,)`` and represents the estimated crowd count
            for each sample, and ``density_map`` contains the per-pixel density estimate.

    Raises:
        RuntimeError: If the model cannot process the provided tensor layout or if the
            device-specific autocast path is unsupported in the current environment.
    """
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
