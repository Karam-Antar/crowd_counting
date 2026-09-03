# ADR 0026: Remove 10% of Manually Added Negative Samples

## Status
Accepted

## Context
The dataset included manually added negative samples (images with no crowds or very few people) to help the model learn to correctly identify empty scenes. Initially, we had added 30-45 negative samples to the training data.

However, during analysis and experimentation, we observed that the model was exhibiting underestimation - predicting fewer people than actually present in images, particularly in crowded scenes. This suggested the model might be biased toward predicting lower crowd counts due to having too many negative (low-count) samples in the training set.

## Decision
We removed 10 samples out of the 30-45 manually added negative samples, reducing the proportion of negative examples in the dataset while still maintaining some negative samples for the model to learn from.

This adjustment was aimed at reducing the model's bias toward underestimation while preserving the ability to recognize empty or sparse scenes.

## Consequences

### Results
✅ **The fix worked effectively** - Removing these 10 negative samples significantly improved the model's performance:
- **Reduced underestimation bias** - The model now predicts crowd counts that are much closer to ground truth
- **Better calibration** - The model learned to predict more realistic crowd sizes across various scenarios
- **Maintained negative sample learning** - The remaining negative samples still allow the model to recognize empty/sparse scenes

### Technical Impact
- Better balance between learning negative examples and avoiding overemphasis on low-count predictions
- Improved generalization across the full range of crowd densities in the dataset
- More accurate predictions on validation and test sets

## Lessons Learned
1. Class balance and sample composition directly affect model prediction bias
2. Even modest adjustments to the negative sample ratio can have significant impact on underestimation issues
3. Underestimation bias should be monitored during training and addressed through dataset composition adjustments

## Related Issues
- Model bias and calibration
- Negative sample handling in crowd counting
