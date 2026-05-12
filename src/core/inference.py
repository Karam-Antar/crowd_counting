import torch
from src import config


@torch.no_grad()
def predict(model, x: torch.Tensor, device=config.device):
        """
        Custom inference method for single inputs.
        """
        """
        Custom inference method for single inputs.
        """
        # 1. Handle dimensionality (C, H, W) -> (1, C, H, W)
        if isinstance(model, torch.nn.Module):
            model.eval()
        if x.ndimension() == 3:
            x = x.unsqueeze(0)

        # 2. Prepare input and run inference
        x = x.to(device)
        logits = model(x)
        
        # 3. Process outputs
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        
        class_idx = int(pred.item())
        class_name = config.CLASS_NAMES[class_idx]
        
        return class_name, class_idx, conf.item()