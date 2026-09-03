# Evaluation Correction Note: Shanghai Part A Test Set

## Background

During an earlier evaluation, the model was tested against the Shanghai Part A `testset` folder while manually added negative samples were still present in that folder. Those negative samples were intended to support training and attention-head learning, but they were not part of the standard Shanghai Part A test set.

This made the resulting evaluation inconsistent with the standard benchmark dataset. The preliminary evaluation produced:

- **MAE:** 65

That value should not be treated as the final benchmark result because the test set contained additional manually added samples.

## Correction

The manually added negative samples were removed from the test set, and the model was evaluated again on the standard Shanghai Part A test data.

The corrected final evaluation produced:

- **MAE:** 68.78
- **RMSE:** 113.93
- **MBE:** -1.71

The corrected result is the value that should be used when describing the final model performance.

## Interpretation

The difference between the preliminary and corrected results is an evaluation-data issue, not evidence that the model changed between the two evaluations. Results from different test-set compositions are not directly comparable, even when the model checkpoint and evaluation code are unchanged.

This note is kept separately from the project README so that the README can present the final result clearly while the documentation still preserves the evaluation mistake and its correction for reproducibility.

## Reproducibility rule

Future evaluations should:

1. Keep manually added negative samples in the training data only unless a separate evaluation protocol explicitly includes them.
2. Use the standard Shanghai Part A test set for benchmark reporting.
3. Record any changes to test-set composition alongside the reported metrics.
4. Avoid comparing metrics from different test-set compositions as if they came from the same benchmark.
