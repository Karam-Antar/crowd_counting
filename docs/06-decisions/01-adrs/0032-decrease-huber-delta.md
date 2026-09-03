# ADR 0032: Decrease Huber delta

## Status

Accepted

## Context

After scaling the composite loss terms and adding count loss, some training curves remained unstable. We investigated `huber_delta`, which controls how long Huber loss behaves quadratically before switching to its linear region.

Increasing `huber_delta` made other loss terms, especially focal loss, suffer during training. This looked as if there were a mathematical conflict between the squared-error behavior of Huber loss and focal loss. The exact mechanism is not yet clear, so this observation should be treated as an empirical hypothesis rather than a proven explanation.

## Decision

Decrease `huber_delta` so that Huber loss enters its linear, absolute-error-like regime for more of the errors encountered during training. With a smaller delta, Huber loss behaves more like L1 loss while retaining its smooth quadratic behavior for very small errors.

## Result

Decreasing `huber_delta` made training more stable. Most loss and metric curves became clearly steadier, while increasing the value caused focal loss and other parts of the composite objective to become less stable or degrade.

The improvement is consistent with reducing the influence of the squared-error region, but the interaction between Huber's quadratic region and focal loss has not been isolated or formally explained.

## Consequences

- Huber loss contributes more like absolute error than squared error over the practical error range.
- Most training curves show clearer stability with the lower delta.
- Focal loss is less likely to suffer from the stronger quadratic Huber behavior observed with higher delta values.
- `huber_delta` must be interpreted relative to the label-scaled density-map values, which can be on the order of `1000` during training.
- The lower delta may reduce pressure for highly precise small residuals, so it should be validated together with density quality, count accuracy, and mask metrics.

## Rationale

For this task, a lower transition point limits the duration and influence of Huber's squared-error regime. That reduced competition between loss terms in the observed experiments and produced more stable optimization. The reason focal loss is particularly affected remains an open question for further investigation.

## Related Decisions

- **ADR 0021**: Replace MSE with Huber loss in density map regression.
- **ADR 0030**: Try count loss with SSIM and focal loss.
- **ADR 0031**: Put more effort into scaling losses and hyperparameters.
