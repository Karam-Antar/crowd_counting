# ADR 0026: Add Custom Metrics MBE and Positive NAE

## Status
Accepted

## Context
During model training and evaluation, we relied on standard metrics like MAE (Mean Absolute Error) and other common loss functions. However, we encountered numerical stability issues with certain evaluation metrics:

1. **Mathematical errors in standard NAE** - The standard Normalized Absolute Error (NAE) metric calculates for all samples, including those with ground truth values of zero or very close to zero, leading to:
   - Division by zero errors
   - Division by extremely small numbers, causing abnormally large values in logs
   - Unstable metric behavior that obscured actual model performance

2. **Lack of directional bias visibility** - Standard error metrics show magnitude of error but don't reveal whether the model tends to systematically underestimate or overestimate crowd counts

## Decision
We implemented two custom metrics:

### 1. Positive NAE (Non-zero NAE)
A modified version of Normalized Absolute Error that:
- **Only operates on samples where ground truth prediction is not zero**
- Avoids division by zero and division by near-zero values
- Provides numerically stable and meaningful normalized error measurements
- Eliminates spurious huge numbers in training logs caused by division by very small denominators

### 2. MBE (Mean Bias Error)
A directional error metric that:
- Tracks systematic bias in predictions (underestimation vs. overestimation)
- Calculated as the mean of (prediction - ground_truth), preserving sign
- Positive MBE indicates the model tends to **overestimate** crowd counts
- Negative MBE indicates the model tends to **underestimate** crowd counts
- Complements absolute error metrics by showing prediction direction bias

## Consequences

### Benefits
✅ **Numerical Stability** - Positive NAE eliminates mathematical errors in metric calculation
✅ **Cleaner Logs** - Training logs now show reasonable, interpretable values without spurious large numbers
✅ **Bias Monitoring** - MBE provides clear visibility into model systematic bias (over/under-estimation)
✅ **Better Debugging** - Directional bias information helps identify and address model calibration issues
✅ **Production Insights** - Understanding if the model tends to underestimate or overestimate is crucial for real-world applications

### Implementation Details
- Positive NAE excludes any samples where ground truth < threshold (e.g., 0 or very small epsilon)
- MBE is a simple signed error metric that's easy to interpret and track
- Both metrics are computed alongside standard metrics for comprehensive evaluation

## Related Issues
- Model calibration and bias tracking
- Numerical stability in metric computation
- Training log quality and interpretability
