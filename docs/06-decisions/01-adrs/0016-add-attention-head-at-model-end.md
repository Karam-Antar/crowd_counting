# ADR 0015: Add attention head at model end to address false positives

## Status

Accepted (with caveat: requires loss function adaptation)

## Context

During evaluation, the model exhibited a **catastrophic false positive problem**: predicting high crowd density in image regions that contained no crowds (backgrounds, walls, empty spaces, hard negative patterns). This was especially severe on:
- Textureless backgrounds
- Repetitive patterns (tiles, grids)
- Shadows and lighting artifacts
- Scene edges and occlusions

The false positives inflated the NAE (Normalized Absolute Error) and MAE metrics and degraded practical usability.

## Decision

Add an attention head at the model output to learn which regions are likely to contain crowds versus background. The attention head acts as a learned mask that suppresses predictions in non-crowd regions and focuses density estimates on actual crowd areas.

## Initial Approach and Limitation

The attention head was implemented as an additional output branch from the encoder-decoder that produces a binary or soft mask:
- 1.0 in regions with crowds
- 0.0 or near-0 in background regions

However, **the initial implementation did not yield expected improvements** because:

1. **Loss function mismatch**: The base loss function (MSE or mask-MSE-SSIM) was optimized only for density map accuracy. It did not explicitly penalize false positives in background regions.

2. **Insufficient supervision**: The model had no strong signal during training that it should suppress density in non-crowd areas. The attention head was initialized randomly and had no supervision beyond the density target.

3. **Conflicting objectives**: The density loss encouraged the model to produce any positive predictions, while the attention head tried to suppress predictions—without a unified loss function, the optimization became confused.

## Consequences

- **Added architectural complexity**: The model now outputs both a density map and an attention mask, requiring careful fusion during inference.
- **Required loss function redesign**: To make the attention head effective, the loss function must explicitly reward background suppression and penalize false positives (e.g., via a mask loss term with focal weighting for background regions).
- **Positive signal for next iteration**: The attention head provided a structural foundation for background/foreground separation, but its success required complementary loss engineering.

## Rationale

The false positive problem is a **misalignment between the learning objective and the desired behavior**:

1. **MSE on dense map alone is insufficient**: MSE treats all pixels equally. A false positive in a background region has the same magnitude loss as a miss in a crowd region, so the model has no incentive to suppress background noise.

2. **Need for negative sample learning**: The model must explicitly learn what **not** to predict (negative samples: empty backgrounds, hard negatives). This requires:
   - A loss term that penalizes high density in background regions
   - Supervision for the attention head (e.g., a binary mask ground truth)
   - Balanced sampling of background-heavy and crowd-heavy regions during training

3. **Attention head is necessary but not sufficient**: The architectural addition (attention head) is a good structure, but the learning signal (loss function) must be redesigned to train it effectively.

## Related Work and Path Forward

See **ADR 0003** (Remove mislabeled data) for the initial approach to handling negative samples through data cleaning. The attention head extends this by learning to suppress false positives during inference, but this requires:
- A composite loss function that includes a mask loss term (penalty for false positives in background)
- Ground truth masks or derived masks from positive/negative sample labels
- Possibly focal loss weighting to emphasize hard negatives

## Related Decisions

- **ADR 0003** (Remove mislabeled data): Addressed data quality issues; attention head addresses model's difficulty learning from negative samples.
- **ADR 0008** (Encoder-decoder architecture): The encoder-decoder's skip connections provide multi-scale spatial information useful for attention refinement.
- **ADR 0005** (Params object): The attention head structure and its weighting can be parameterized via `BaseParams` (e.g., `attention_weight`, `attention_loss_type`).
