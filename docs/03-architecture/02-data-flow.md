# Data Flow

The main data flow is:

1. image and target pairs are loaded from the configured dataset folders,
2. paired files are aligned by ordering,
3. per-sample transforms are applied,
4. batch-level padding is applied for compatibility with the model,
5. the model produces a density prediction,
6. evaluation compares the prediction against the ground truth,
7. the run is logged to MLflow.
8. if the model is the best historically based on specific metric it gets automatically logged to mlflow-registry

This flow is central because many project issues are caused by hidden assumptions in the dataset contract rather than the model itself.
