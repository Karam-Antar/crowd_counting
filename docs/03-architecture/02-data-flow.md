# Data Flow

The main data flow is:

1. image and target pairs are loaded from the configured dataset folders,
2. paired files are aligned by ordering,
3. per-sample transforms are applied,
4. batch-level padding is applied for compatibility with the model,
5. the model produces a density prediction,
6. evaluation compares the prediction against the ground truth,
7. the run is logged to MLflow.
8. the configured tracker records parameters and metrics, and the registry can persist the resulting model payload.

During training, image and density-map transforms are applied jointly for spatial operations. Images are converted and normalized independently; density maps are scaled by the training label scaler. Validation and test data use deterministic preprocessing, dynamic batch padding, and optional fixed five-crop evaluation.

The model returns density information and, in the attention-head configuration, a foreground confidence map. Soft sigmoid gating combines the two before the final convolution. Counts are calculated from the resulting density map by summation with the appropriate scaling reversal.

Training orchestration is shared by `BaseExperimentRunner`; `StandardRunner` executes a fixed run and `OptunaTuner` creates trial-specific parameters and tracking contexts. A model payload can contain the trained model, parameters, tracker information, and metrics for registry upload.

This flow is central because many project issues are caused by hidden assumptions in the dataset contract rather than the model itself.
