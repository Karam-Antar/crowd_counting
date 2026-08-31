# ADR 0007: Give up on HRNet

## Status

Superseded

## Context

HRNet (High-Resolution Network) is an encoder-decoder architecture designed to maintain high-resolution feature maps throughout the network. It was evaluated as a backbone for the crowd-counting model to improve spatial detail preservation and localization accuracy.

## Decision

Discontinue using HRNet as the primary backbone. While the model showed promise initially, it reached a performance plateau at val_nae ≈ 0.20 and required substantial computational resources (training time, GPU memory) compared to the marginal improvements achieved.

## Consequences

- **Freed compute budget**: HRNet's high memory footprint and training time are redirected to other architectures (EfficientNet, ResNet) that achieve better results with lower overhead.
- **Faster experiment iteration**: Removing a computationally expensive baseline accelerates hyperparameter tuning cycles.
- **Clearer focus**: The project now prioritizes efficient backbones that scale well and deliver consistent improvements.

## Rationale

HRNet's design prioritizes spatial detail at the cost of computational efficiency. In practice, the additional detail did not translate to better crowd-counting performance; the model plateaued and further tuning yielded diminishing returns. Given the computational budget constraints and the availability of lighter architectures that achieve comparable or better results, continuing to invest in HRNet was not justified.

This is a common pattern in deep learning: architecturally sophisticated models do not always outperform simpler alternatives on every task, especially when computational overhead is considered.

## Related Decisions

- **ADR 0004** (Class inheritance): The `backbone` parameter in `BaseParams` supports multiple architecture choices; HRNet is simply not selected in current experiments.
- **ADR 0006** (Give up coordinate attention): Similarly, architectural innovations require empirical validation; when they don't pay off, it is better to move on.
