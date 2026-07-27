import torchmetrics
import torch

class MeanBiasError(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        # State variables to accumulate across the whole epoch
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Crucial: Pred - Target (Preserves the sign)
        self.sum_error += torch.sum(preds - target)
        self.total += target.numel()

    def compute(self):
        return self.sum_error / self.total


class PositiveOnlyNAE(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_nae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Create a boolean mask of images that actually have people
        valid_mask = target > 0
        
        # If the batch has no positive samples, skip it
        if not valid_mask.any():
            return
            
        valid_preds = preds[valid_mask]
        valid_target = target[valid_mask]
        
        # Now division by zero is mathematically impossible
        self.sum_nae += torch.sum(torch.abs(valid_preds - valid_target) / valid_target)
        self.total += valid_target.numel()

    def compute(self):
        # Prevent division by zero if the entire epoch had no positive samples (unlikely)
        if self.total == 0:
            return torch.tensor(0.0)
        return self.sum_nae / self.total