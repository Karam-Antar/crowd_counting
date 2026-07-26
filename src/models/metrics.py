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