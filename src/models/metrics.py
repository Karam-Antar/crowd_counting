import torchmetrics
import torch

class MeanBiasError(torchmetrics.Metric):
    """Accumulate mean signed error across a validation or training epoch."""
    def __init__(self):
        """Initialize the sum-based metric state required for epoch-level aggregation."""
        super().__init__()
        # State variables to accumulate across the whole epoch
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Accumulate the signed prediction error for the current batch.

        Args:
            preds (torch.Tensor): Predicted count values.
            target (torch.Tensor): Ground-truth count values.
        """
        # Crucial: Pred - Target (Preserves the sign)
        self.sum_error += torch.sum(preds - target)
        self.total += target.numel()

    def compute(self):
        """Return the mean bias error across all accumulated batches.

        Returns:
            torch.Tensor: Mean signed error value.
        """
        return self.sum_error / self.total


class PositiveOnlyNAE(torchmetrics.Metric):
    """Compute normalized absolute error only on samples with a positive target count."""
    def __init__(self):
        """Initialize the positive-only NAE metric state."""
        super().__init__()
        self.add_state("sum_nae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Accumulate the normalized absolute error for positive-count samples.

        Args:
            preds (torch.Tensor): Predicted counts.
            target (torch.Tensor): Ground-truth counts.
        """
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
        """Return the positive-only normalized absolute error.

        Returns:
            torch.Tensor: Average NAE over positive-count samples.
        """
        # Prevent division by zero if the entire epoch had no positive samples (unlikely)
        if self.total == 0:
            return torch.tensor(0.0)
        return self.sum_nae / self.total



class CombinedMAEMBE(torchmetrics.Metric):
    """Combine MAE and mean bias error into a single differentiable error metric."""
    # Set higher_is_better=False since this is an error metric
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        """Initialize the accumulators required for MAE + MBE aggregation."""
        super().__init__()
        # add_state handles cross-GPU syncing automatically in PyTorch Lightning
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Accumulate absolute and signed errors for each sample in the batch.

        Args:
            preds (torch.Tensor): Predicted counts.
            target (torch.Tensor): Ground-truth counts.
        """
        # Flatten tensors to handle potential batch dimensions easily
        preds = preds.view(-1)
        target = target.view(-1)
        
        # Calculate raw error
        error = preds - target
        
        # Accumulate metrics for the batch
        self.sum_abs_error += torch.sum(torch.abs(error))
        self.sum_error += torch.sum(error)
        self.total += target.numel()

    def compute(self):
        """Return the combined MAE + |MBE| score for the epoch.

        Returns:
            torch.Tensor: Combined error metric value.
        """
        # Calculate final MAE and MBE across all accumulated batches
        mae = self.sum_abs_error / self.total
        mbe = self.sum_error / self.total
        
        # Return the combined score
        return mae + torch.abs(mbe)