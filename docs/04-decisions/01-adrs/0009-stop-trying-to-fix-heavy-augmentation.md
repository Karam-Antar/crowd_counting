# ADR 0009: Stop trying to fix heavy augmentation

## Status

Accepted

## Context

Data augmentation is widely recognized as a regularization technique that improves model generalization. Early experiments explored aggressive augmentation strategies (high `aug_factor`, large number of `num_ops` augmentation operations) to improve the model's robustness to natural variations in crowd density, lighting, and perspective.

## Decision

Reduce augmentation intensity and stop pursuing heavier augmentation as a primary path to improved model performance. Experiments demonstrated that more augmentation does not reliably translate to better diversity or improved validation metrics. Additionally, heavy augmentation introduced unintended side effects: added training latency and occasional numerical instabilities (e.g., NaN values in loss computation, gradient anomalies).

## Consequences

- **Simplified training pipeline**: Reduced augmentation overhead lowers training time and memory footprint.
- **Improved numerical stability**: Fewer edge-case augmentations eliminate unexpected gradient issues or loss spikes.
- **Clearer experiment signals**: With moderate augmentation, validation curves are cleaner and hyperparameter effects are easier to isolate.
- **Redirected optimization effort**: Energy previously spent tuning augmentation intensity is now focused on loss functions, learning rates, and architecture tuning.

## Rationale

Heavy augmentation often suffers from diminishing returns and can introduce artifacts:

1. **Limited diversity benefit**: Augmentation generates synthetic variations, but for crowd density prediction, excessive transforms may create unrealistic or contradictory samples (e.g., geometric warping that distorts the spatial structure of crowds without corresponding label adjustments).

2. **Numerical instability**: Aggressive augmentation (random crops, color jittering, elastic deformations) can produce edge cases during batch normalization or loss computation, especially when labels are density maps that depend on spatial alignment.

3. **Training latency**: Heavy augmentation pipelines consume CPU cycles per-batch, increasing epoch time and reducing the number of experiments that can run in a fixed compute budget.

Empirical results showed that moderate augmentation (fewer operations, lower intensity) paired with better loss functions and architecture choices outperformed heavy augmentation strategies. This is consistent with the observation that augmentation is one lever among many; it is not a universal cure for generalization.

## Related Decisions

- **ADR 0005** (Params object): The `aug_factor` and `num_ops` parameters in `BaseParams` control augmentation intensity; they are now kept conservative or left at default.
- **ADR 0008** (Encoder-decoder architecture): The improved baseline reduced pressure to over-augment; better feature extraction eliminated the need for aggressive regularization.
