# ADR 0022: Try dynamic loss weighting

## Status

Deferred

## Context

The composite loss function combines multiple terms with fixed weights:

$$\mathcal{L} = \lambda_{density} \cdot \mathcal{L}_{Huber+SSIM} + \lambda_{mask} \cdot \mathcal{L}_{focal}$$

Choosing the right balance between $\lambda_{density}$ and $\lambda_{mask}$ is difficult:
- Too high $\lambda_{density}$: Density predictions improve, but false positives persist
- Too high $\lambda_{mask}$: False positives suppressed, but density structure degrades

The hypothesis was that these weights could be learned dynamically during training, allowing the model to automatically adjust the balance as it progresses.

## Decision

Implement dynamic loss weighting where $\lambda_{density}$ and $\lambda_{mask}$ are learned parameters or scheduled adaptively based on loss values during training. Early attempts included:
- **Learnable weights**: Adding trainable scalar parameters that adjust weights
- **Adaptive scheduling**: Adjusting weights based on the ratio of loss terms across batches
- **Uncertainty weighting**: Using learned task uncertainties to weight loss components

## Result: No Clear Improvement

Dynamic loss weighting did not produce measurable improvements over fixed weights. Training dynamics became less stable, and the model's convergence behavior was less predictable.

## Issues Encountered

1. **Unstable optimization**: Dynamically adjusting loss weights introduced additional sources of variance, making convergence less stable.

2. **Insufficient tuning**: The dynamic weighting schemes attempted were preliminary and lacked the systematic hyperparameter exploration needed for success.

3. **Complexity-benefit trade-off**: The added complexity of dynamic weighting did not justify the performance gains (which were minimal or absent).

## Path Forward

Dynamic loss weighting is a promising direction but requires:
- More careful design of the scheduling/adaptation mechanism
- Better initialization and constraint strategies
- Systematic hyperparameter tuning for any dynamic component
- Potentially more sophisticated techniques (e.g., gradient-based meta-learning, multi-task learning frameworks)

Fixed weights with careful manual tuning via hyperparameter search (e.g., Optuna) proved more reliable and effective than preliminary dynamic approaches.

## Related Decisions

- **ADR 0005** (Params object): Fixed weights (`mask_loss_weight`, `ssim_weight`) are controlled via `BaseParams`.
- **ADR 0021** (Replace MSE with Huber): Proper loss function design reduces need for complex weighting schemes.
- **ADR 0020** (Replace BCE with focal loss): Better loss function design is more impactful than weighting tricks.
