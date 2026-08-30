# Run Comparison Guide

When comparing runs, do not rely on the metric numbers alone.

Always check:

- whether the dataset changed,
- whether preprocessing changed,
- whether the split strategy changed,
- whether the model code changed,
- whether label scaling or augmentation changed.

Many experiments in this project look different because the dataset or training contract changed, not because the model itself improved.
