# ADR 0006: Give up on coordinate attention

## Status

Superseded

## Context

Coordinate attention (CoordAtt) is a lightweight attention mechanism designed to capture cross-channel relationships along spatial coordinates. It was implemented in the decoder as `decoder_attention_type='coord_att'` to improve the model's ability to localize crowds in high-density scenes.

## Decision

Remove coordinate attention from the decoder. The mechanism added computational overhead without measurable improvement to model performance on the validation metrics (MAE, NAE, RMSE).

## Consequences

- **Simplified decoder**: Removes conditional logic for attention type selection.
- **Reduced memory and compute**: Faster training and inference without the coordinate attention overhead.
- **Cleaner experiment history**: Prevents future trials from exploring a dead-end architecture direction.
- **Baseline stability**: Simplifies the default decoder configuration.

## Rationale

Coordinate attention incurred a non-negligible per-layer cost (extra channel and spatial projections) without yielding better validation results in practice. This is a common occurrence in architecture search: theoretically motivated components do not always translate to empirical gains.

Removing it frees up compute budget for other improvements (e.g., larger batch sizes, more epochs, better loss functions) and prevents future experiments from wasting trials on the same dead end.

## Related Decisions

- **ADR 0004** (Class inheritance): The `decoder_attention_type` parameter is part of `BaseParams`, but is typically left as `None` after this decision.
- **ADR 0002** (Geometry-adaptive targets): Attention mechanisms were explored as one approach to handle scale variation, but were superseded by better target design.
