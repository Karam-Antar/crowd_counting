# ADR 0030: Try count loss with SSIM and focal loss

## Status

Accepted

## Context

The training objective combined density-map reconstruction with SSIM and focal loss. The density component used point-wise Huber loss, which provided local pixel-level supervision but did not directly constrain the aggregate crowd count. As a result, the MBE metric was very unstable during training, making systematic overestimation or underestimation difficult to assess.

## Decision

Replace the point-wise Huber component with an overall-count Huber loss while retaining SSIM and focal loss. The count loss is computed from the difference between the predicted total count and the ground-truth total count for each image, giving the model a direct signal for aggregate count accuracy.

## Consequences

- The MBE metric showed clear stability after adding the count loss; it had been very unstable with the point-wise Huber formulation.
- The objective directly penalizes errors in the total predicted count, improving the usefulness of MBE as a training and evaluation signal.
- SSIM and focal loss continue to provide structural and mask-related supervision alongside the global count constraint.
- Because count loss does not specify where density should be placed, it must remain paired with spatial and structural losses rather than replacing them entirely.

## Rationale

The observed stabilization of MBE indicates that the overall-count Huber term improved control of systematic count bias. This makes the combined objective a better fit for crowd counting, where aggregate count accuracy matters in addition to pixel-level density-map quality.

## Related Decisions

- **ADR 0021**: Replace MSE with Huber loss in density map regression.
- **ADR 0026**: Add custom metrics MBE and Positive NAE.
