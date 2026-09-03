# ADR 0014: Try ensembled model prediction

## Status

Rejected

## Context

Model ensembling is a classic technique to improve robustness and accuracy by combining predictions from multiple models trained with different initializations or hyperparameters. The approach was explored to boost performance by averaging the outputs of several crowd-counting models.

## Decision

Abandon model ensembling. While ensembles can improve accuracy, the approach incurred prohibitive costs: massive memory overhead and severe inference latency. A single, well-tuned model with better architecture and loss function is a superior path.

## Consequences

- **Memory overhead**: Storing and loading multiple model copies requires N times the memory of a single model, making deployment on resource-constrained environments infeasible.
- **Inference latency**: Ensembles require N forward passes; inference time scales linearly with ensemble size, making real-time or near-real-time predictions impractical.
- **Deployment complexity**: Serving an ensemble requires coordinating multiple model instances, increasing operational burden.

## Rationale

Ensembles are a form of "brute-force" accuracy improvement: instead of building a better single model, ensemble methods combine weaker predictors. However, in this project:

1. **Diminishing returns**: Each additional model contributes less to the ensemble's accuracy, but the memory and latency cost is linear.

2. **Single powerful model is better**: The computational resources spent on multiple models can be redirected to:
   - Better neural network architectures (e.g., exploring larger decoder depths, richer skip connections)
   - Better loss functions (e.g., composite losses that balance density and boundary accuracy)
   - More extensive hyperparameter tuning on a single model
   - Larger batch sizes and longer training on a single model

3. **Fundamental inefficiency**: Ensembles mask the real problem—that the underlying model is not performing well enough. Solving the root cause (architecture, loss, training) is more sustainable.

## Alternative Path

Instead of ensembling, improve the single model:
- Refine the encoder-decoder architecture (decoder depth, feature fusion strategies)
- Experiment with composite loss functions (combining density MSE, boundary information, SSIM)
- Apply advanced regularization (dropout, batch normalization tuning, weight decay)
- Increase training compute on fewer, better-designed models

A single powerful model is:
- Simpler to deploy
- Faster to inference
- More memory-efficient
- More maintainable
- Easier to debug and understand

## Related Decisions

- **ADR 0008** (Encoder-decoder architecture): The foundation for building a powerful single model.
- **ADR 0009** (Stop heavy augmentation): Focus on architecture and loss rather than circumventing poor design with regularization tricks.
