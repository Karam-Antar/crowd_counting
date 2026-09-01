# ADR 0025: Add Batching Ability to Validation and Test DataLoaders

## Status
Accepted

## Context
Previously, validation and test dataloaders processed samples one at a time without batching. This meant that validation runs after each training epoch were performed sequentially on individual samples, which could be slower than necessary.

There is potential to speed up validation by using batch sizes larger than 1, since batching allows for vectorized operations and better GPU utilization.

## Decision
We implemented a custom `collate_fn` for validation and test dataloaders that:

1. **Pads samples to match the largest sample in the batch** - All samples in a batch are resized to the dimensions of the largest sample within that batch
2. **Ensures dimensions are multiples of 32** - After padding to the batch maximum, dimensions are further padded to become multiples of 32 to satisfy neural network architecture constraints

This approach enables efficient batched processing of validation and test data.

## Consequences

### Benefits
- **Faster validation**: Using batch sizes larger than 1 enables vectorized operations and improved GPU utilization during validation runs
- **Improved throughput**: Validation at the end of each epoch can run faster with proper batching

### Tradeoffs and Risks

⚠️ **Memory Consumption**: Users should exercise caution when selecting batch sizes, especially with datasets containing highly variable image sizes. 

- All samples in a batch are **padded to match the size of the largest sample** in that batch
- This can lead to significant memory overhead if the batch contains even one large image
- **Example**: In datasets like JHU-Crowd++, images can be as large as 4000×4000 pixels. If a batch of 32 samples includes one such large image, all other samples will be padded to 4000×4000, even if most images are significantly smaller, resulting in massive memory consumption
- Memory usage scales with the square of the largest dimension, so large outliers can have extreme impact

### Recommendations
- Start with small batch sizes (e.g., 1-4) for validation to ensure stability
- Monitor GPU memory usage when increasing batch sizes
- Be especially cautious with datasets that contain large outliers
- Consider dataset characteristics when selecting batch size for production use

## Related Issues
- Validation performance optimization
- Memory efficiency in batched processing
