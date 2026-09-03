# ADR 0019: Add spatially weighted loss

## Status

Rejected

## Context

The model struggled with false positives in background regions (ADR 0015). Rather than using an explicit attention head mask, an alternative approach was to weight the loss differently across the image: penalizing errors more heavily in background regions and less heavily in obvious crowd regions.

The hypothesis was that by spatially weighting the MSE loss based on the ground-truth density distribution, the model would learn to suppress predictions in empty areas without needing a separate attention head branch.

## Decision

Implement a spatially weighted loss function:

$$\mathcal{L}_{weighted} = \sum_{i,j} w_{i,j} \cdot (y_{i,j} - \hat{y}_{i,j})^2$$

where the weight $w_{i,j}$ is derived from the ground-truth density:
- **High weight** in low-density (background) regions: penalize false positives more harshly
- **Low weight** in high-density (crowd) regions: allow some magnitude error in difficult crowd areas
- Weight formula: $w_{i,j} = \frac{1}{1 + y_{i,j}} $ or similar inverse-density function

The goal was to make false positives in empty space more costly than under-predictions in crowded areas.

## Result: Minimal Impact (Slight Degradation)

The spatially weighted loss did not produce the expected improvement. Validation metrics (NAE, MAE) either remained unchanged or slightly degraded. The model did not learn to suppress false positives as intended.

## Why It Failed

1. **Inverted supervision signal**: Weighting background regions higher actually **penalizes** high-magnitude errors in empty space. However, this is backward—the model's false positive problem is that it **predicts non-zero density** in background regions, not that it has high magnitude errors when forced to predict there.


2. **Lack of explicit binary signal**: Unlike the attention head (which outputs 0 or 1 for background/foreground), spatial weighting is continuous and ambiguous. The model cannot clearly distinguish whether a small prediction should be suppressed or tolerated.

## Lessons Learned

- **Loss weighting alone is insufficient for binary decisions**: The false positive problem is fundamentally about distinguishing background from crowd (binary), not about continuous magnitude weighting.
- **Negative supervision requires explicit targets**: Penalizing false positives requires a ground-truth target (e.g., a binary mask) that makes the negative space explicit. Weighting the loss is an indirect proxy that doesn't communicate this clearly.
- **Implicit vs. explicit supervision**: The attention head (ADR 0015) succeeds because it explicitly predicts foreground vs. background. Spatial weighting is too implicit.

## Why the Attention Head Approach Is Better

The attention head addresses the root problem directly:
- **Explicit output**: Sigmoid mask (0–1) for each pixel
- **Direct supervision**: BCE loss on ground-truth binary mask
- **Clear gradient**: The model learns what background pixels should output (close to 0)

Spatial weighting, by contrast, only indirectly encourages background suppression without making it a primary learning objective.

## Why This Approach Was Tried

Spatial weighting seemed appealing because:
- No additional model branch (architecturally simple)
- Theoretically sound (weight errors by importance)

However, the crowd-counting task requires a stronger signal than loss weighting alone can provide.

## Related Decisions

- **ADR 0015** (Add attention head): The architecturally explicit approach that works better than spatial weighting.
- **ADR 0017** (Add BCE loss): Provides the explicit binary supervision that spatial weighting lacked.
- **ADR 0001** (Density-map regression): The primary target is density, not attention; indirect weighting is insufficient.
