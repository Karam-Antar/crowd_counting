# ADR 0031: Put more effort into scaling losses and hyperparameters

## Status

Accepted

## Context

After adding count loss, some training curves were still unstable. The instability was caused by differences in the natural scales of the loss terms. Count loss can have values in the hundreds, while SSIM and focal loss are typically less than one. Adding count loss made this existing scale mismatch wider and allowed the count term to dominate the combined objective unless it was scaled deliberately.

## Decision

Scale each loss using a weight made of two separate factors:

1. **Scale-unification factor**: brings the loss term into a comparable numerical range with the other terms.
2. **Penalty factor**: controls how strictly the model is penalized for that type of error after the scales have been aligned.

For example, the training configuration uses:

```python
ssim_weight=1 * 2.8
mse_weight=0.000001 * 1.5
mask_loss_weight=10 * 0.4
```

The very small first factor for `mse_weight` is needed because count errors can be in the hundreds. It does not mean that count accuracy is unimportant. The second factor, `1.5`, is the actual penalty multiplier and means that count errors are penalized more strongly than the normalized baseline. Similarly, the first factor for each other loss unifies its scale, while the second factor expresses its relative importance.

We also adapted `huber_delta` to the mean scale of the dataset and to the scale of density-map counts after label scaling during training. The relevant count scale is usually around `1000`, so the Huber transition must be chosen in that scaled space rather than copied from unscaled count values.

## Consequences

- Loss terms contribute to optimization on comparable numerical scales, reducing instability in training curves.
- The normalized loss weights make the distinction between numerical scaling and modeling preference explicit.
- Count loss can remain important without overwhelming SSIM or focal loss because its scale and penalty strength are tuned independently.
- `huber_delta` is now interpreted consistently with the scaled density-map and count values used during training.
- Loss weights and `huber_delta` require tuning when label scaling, dataset statistics, or the loss formulation changes.

## Rationale

The goal is not to make all losses identical. The first factor prevents a loss from dominating only because of its units or magnitude, while the second factor preserves the intended optimization priority. Separating these concerns makes the composite objective easier to reason about and gives hyperparameter tuning a meaningful penalty-strength parameter for each error type.

## Related Decisions

- **ADR 0021**: Replace MSE with Huber loss in density map regression.
- **ADR 0022**: Try dynamic loss weighting.
- **ADR 0030**: Try count loss with SSIM and focal loss.
