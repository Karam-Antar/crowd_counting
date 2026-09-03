# ADR 0028: Add Five Crops for Validation

## Status
Accepted (with Future Recommendations)

## Context
Validation on large datasets with large images (e.g., JHU-Crowd++) presents significant challenges:

1. **Memory constraints** - Full validation images in JHU-Crowd++ can be extremely large (up to 4000×4000 pixels), making full-image validation computationally expensive, we needed at least 40 GB VRAM GPU to run training with validation on full resolution validation images and for testing even 40 GB were not enough
2. **Validation performance** - Running validation on the complete validation set with full-resolution images after each epoch significantly slows down training
3. **Cropping strategy trade-offs**:
   - **Random cropping**: Inconsistent crops across epochs would invalidate metric comparisons and introduce noise
   - **Resizing**: Would distort the ground-truth density maps (since resizing image with resized density map causes alignment and accuracy issues)

## Decision
We implemented a **five-crop validation strategy**:

### Approach
- Extract **5 fixed, non-overlapping crops** from each validation image
- Crops are **consistently positioned** at the same locations for each image (not randomized)
- Crops are **strategically distributed** across the image to capture general patterns and variations
- All crops maintain **consistent size** to enable fair metric comparison across validation runs

### Crop Locations
Five crops are taken from distributed regions of the image to ensure diverse spatial coverage and represent the overall characteristics of the full image without requiring full-resolution processing.

## Consequences

### Benefits
✅ **Reduced Memory Usage** - Validating on crops instead of full images significantly decreases GPU/CPU memory requirements
✅ **Faster Validation** - Validation runs complete quicker, reducing per-epoch training time
✅ **Maintained Consistency** - Fixed crop locations enable valid metric comparisons across epochs and runs
✅ **Preserved Diversity** - Distributed crop placement captures different regions of the image, reflecting overall patterns
✅ **Scalability** - Makes validation feasible on very large images without compromising model evaluation quality

### Why Not Other Approaches
❌ **Random cropping**: Would give different crops each validation run, making it impossible to track consistent metrics and introducing artificial noise into validation curves

❌ **Resizing**: Would require resizing both images and ground-truth density maps, which:
- Distorts the density map through interpolation
- Creates misalignment between image and density map
- Compromises both training and validation reliability

## Future Recommendations
⭐ **Expand Five-Crop Strategy** - We recommend expanding the use of five-crop validation in future iterations:
- The strategy proved effective and reliable
- Project timeline constraints limited extensive validation of this approach
- Consider increasing to 10+ crops for even better coverage if memory allows
- Could be combined with ensemble predictions (averaging predictions across crops) for improved robustness

## Implementation Notes
- Five crops per image are processed and metrics are aggregated
- Metrics should be averaged across all crop predictions for consistent comparison
- This approach scales well to large datasets while maintaining validation integrity

## Related Issues
- Validation performance optimization
- Memory efficiency in large-scale validation
- Density map preservation in multi-scale evaluation
