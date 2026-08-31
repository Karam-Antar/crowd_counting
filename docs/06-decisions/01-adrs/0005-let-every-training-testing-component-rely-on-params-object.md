# ADR 0005: Let every training and testing component rely on a shared params object

## Status

Accepted

## Context

The crowd-counting pipeline involves many interconnected components that need consistent configuration:

- **Data pipeline** (`CrowdDataModule`): Image size, crop size, augmentation settings, train/val split
- **Model architecture** (`CrowdCounter`, `EncoderDecoder`): Backbone selection, decoder output channels, dropout rates
- **Training loop** (`BaseLitModel`): Loss function selection, loss weights (SSIM, mask loss), thresholds
- **Optimization** (`BaseLitModel`): Learning rate, learning rate schedule, regularization (L2), gradient clipping, accumulation
- **Experiment orchestration** (`StandardRunner`, `OptunaTuner`): Monitoring metric, early stopping patience, checkpoint management
- **Inference** (`ProductionPyTorchWrapper`): Model architecture and label scaling factor for serving

Historically, scattered configuration might be passed through:
- Separate command-line arguments to each component
- Global configuration files with unclear ownership
- Magic numbers hardcoded in function bodies
- Inconsistent settings between training and inference

This fragmentation causes:
- **Reproducibility issues**: A model trained with one setting may be served with different scaling or preprocessing.
- **Comparison problems**: When comparing experiment runs, it's unclear whether differences come from model changes or configuration drift.
- **Hyperparameter tuning friction**: Optuna trials must manually suggest values and thread them through multiple APIs.
- **Experiment drift**: Over time, the training contract evolves, making historical runs hard to interpret.

## Decision

Create a single `BaseParams` dataclass that encapsulates all model, training, and data configuration. Every training and inference component depends on a `BaseParams` instance passed at initialization.

**Key design decisions:**

1. **Unified configuration container**: `BaseParams` is a `@dataclass` containing ~60 fields organized by concern:
   - Image & data augmentation (image_size, crop_size, aug_factor, five_crops)
   - Architecture tweaks (backbone, model_class, decoder_out_channels, dropout)
   - Training hardware/flow (batch_size, val_batch_size, epochs, stop_patience)
   - Optimization & regularization (lr, l2_reg, lr_schedule, grad_clip, grad_accumulation)
   - Loss configuration (loss_function, ssim_weight, mask_loss_alpha, mask_loss_gamma, huber_delta)
   - Dataset metadata (dataset, train_size)
   - Experiment metadata (suggested_params flag for Optuna-generated configs)

2. **Serialization support**: Every `BaseParams` instance can be converted to/from JSON or dictionaries:
   - `to_json()` / `from_json()`: Persist configuration to disk and reload it
   - `to_dict()` / `from_dict()`: Convert for MLflow logging and experiment comparison
   - Support for flattening (dot-delimited keys) and string coercion for tracking backends

3. **Hyperparameter tuning integration**: The `suggest()` class method generates trial-specific configurations by:
   - Using Optuna's `trial.suggest_*` API for each tunable parameter
   - Respecting domain knowledge (e.g., log-scale for learning rates)
   - Mapping architecture names to unfrozen block counts
   - Returning a fully-formed `BaseParams` instance

4. **Component dependency**: All major components accept and store a `BaseParams` instance:
   - `CrowdDataModule(params)`: Uses image_size, crop_size, batch_size, augmentation settings
   - `CrowdCounter(params)`: Uses backbone, model_class, decoder_out_channels, dropout
   - `BaseLitModel(params)`: Uses loss_function, lr, l2_reg, monitor_metric, and all loss-specific weights
   - `MaskMSESSIMLoss(params)`: Uses ssim_weight, mask_loss_alpha, mask_loss_gamma, huber_delta
   - `BaseExperimentRunner(...)`: Uses params for initialization and passes to model/datamodule
   - `ProductionPyTorchWrapper.load_context()`: Loads params from JSON and reconstructs the model

5. **Experiment tracking integration**: MLflow receives:
   - Flattened params dictionary via `params.to_dict(flatten=True, to_str=True)` in `tracker.log_results()`
   - Serialized JSON artifact of the full params via `params.to_json()` in registry uploads
   - Can compare runs by examining tracked params in MLflow UI

## Consequences

### Positive

- **Single source of truth**: All configuration originates from one object, eliminating confusion about which setting was used.
- **Reproducibility**: Serializing and loading `BaseParams` from JSON ensures exact configuration replay at inference time.
- **Experiment comparison**: MLflow can compare runs using tracked params, making it clear whether differences come from architecture or data.
- **Hyperparameter tuning**: Optuna integration via `suggest()` is seamless; new parameters can be added to the search space without touching runner code.
- **Cleaner APIs**: Components have fewer constructor arguments; they depend on a well-documented `BaseParams` instance.
- **Inference stability**: The `ProductionPyTorchWrapper` loads params from the same JSON file, guaranteeing that label scaling, model architecture, and preprocessing match the training run.
- **Maintainability**: Future contributors can read `BaseParams` fields to understand the full configuration space.
- **Experiment drift detection**: Storing params in MLflow makes it easy to spot when configuration assumptions changed between runs.

### Trade-offs

- **Large dataclass**: With ~60 fields, `BaseParams` is sizable. Developers must understand what each field controls.
- **Optional fields**: Many fields are `Optional[...]` because not all configurations use them (e.g., `decoder_attention_type` is None for most runs).
- **Validation complexity**: No built-in validation for invalid combinations (e.g., a crop_size larger than image_size). Errors may emerge at runtime.
- **Serialization fragility**: Changing the `BaseParams` schema requires migration logic for existing experiment artifacts stored in MLflow.
- **Default values**: Choosing sensible defaults for ~60 parameters is difficult; legacy runs may have used different defaults.

## Rationale

This pattern was chosen because:

1. **ML projects are configuration-heavy**: Crowd-counting experiments involve dozens of tunable settings across data, architecture, and optimization. Scattering them across files or APIs makes it easy to use the wrong value or to lose track of what was actually run.

2. **Reproducibility is critical**: A model trained with label_scaler=1000 cannot be served with label_scaler=1; the mismatch would cause incorrect predictions. By centralizing configuration and versioning it with the model artifact, we prevent this class of bug.

3. **Hyperparameter tuning is inherently parameter-centric**: Optuna works by suggesting parameter values and comparing results. Having a single `suggest()` method in the params class makes the tuning loop natural and maintainable.

4. **Experiment comparison is central to ML workflows**: The project uses MLflow to log hundreds of runs. Having flattened, string-coercible params makes it trivial to compare two experiments in the MLflow UI and ask "why was this run 5% better?"—often the answer is in the params diff.

5. **Future evolution**: The project's training contract will evolve (new loss functions, new architectures, new augmentation strategies). Having a central `BaseParams` class makes adding new fields low-friction; components that don't use a field simply ignore it.

## Related Decisions

- **ADR 0004** (Class inheritance for experiment tracking): The `BaseExperimentRunner` receives a `BaseParams` instance and passes it to all subcomponents, enabling the inheritance hierarchy to work cleanly.
- **ADR 0001** (Density-map regression): The loss functions and label scaling factor are embedded in `BaseParams`, standardizing how density targets are handled.
- **ADR 0002** (Geometry-adaptive targets): The `gt_mask_threshold` and mask loss weights are part of `BaseParams`, ensuring consistent target generation across runs.

## Example Usage

### Training a single model:
```python
from src.core.params import BaseParams
from src.experiment import StandardRunner

params = BaseParams(
    backbone='efficientnet-b4',
    lr=0.0005,
    loss_function='mask_mse_ssim',
    ssim_weight=0.3,
    epochs=100,
    batch_size=16
)
runner = StandardRunner(model_cls=CrowdCounter, tracker=tracker, params=params)
runner.run()
```

### Hyperparameter tuning:
```python
from src.experiment import OptunaTuner

tuner = OptunaTuner(
    experiment='crowd_counting',
    model_cls=CrowdCounter,
    study_name_suffix='exp_v2',
    n_trials=50
)
study = tuner.run()  # Each trial calls BaseParams.suggest(trial) internally
```

### Inference with loaded params:
```python
from src.core.params import BaseParams

params = BaseParams.from_json('model_artifacts/params.json')
model = CrowdCounter(params=params)
model.load_state_dict(...)
prediction = model(image)
count = prediction.sum() / params.label_scaler  # Use same scaler as training
```
