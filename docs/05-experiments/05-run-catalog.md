# Run Catalog

This catalog should track the most important MLflow run metadata:

- run ID,
- dataset version,
- code version or training script,
- key hyperparameters,
- validation metrics,
- model artifacts,
- notes about whether the run was a baseline, candidate, or failure.

A good run catalog makes it much easier to interpret historical work without having to dig through raw MLflow UI records.
