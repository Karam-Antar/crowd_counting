# Branch and Experiment Summary

This document explains how experiment branches, training scripts, and MLflow runs relate to one another. Understanding the history of runs and architectural decisions helps identify why certain configurations were chosen and which models are most reliable.

## Experiment Structure

The project uses MLflow to centralize experiment tracking:

- **Experiment name**: `"crowd_counting"` (default for all standard runs)
- **Tracking backend**: Configured via `MLFLOW_DB_URL` in `.env`
- **Run types**:
  - **Standard runs**: Single training jobs via `StandardRunner` (from `scripts/train.py`)
  - **Tuning runs**: Hyperparameter search via `OptunaTuner` (from `scripts/tune.py`)

## MLflow Run Organization

### Single Training Runs

Each execution of `scripts/train.py` creates an MLflow run that logs:

- **Hyperparameters**: Complete `BaseParams` configuration
- **Metrics**: Validation and training metrics at each epoch
- **Artifacts**: 
  - `checkpoints/` - Best model checkpoint
  - `model_config.json` - Serialized parameters
  - Experiment metadata and run notes
- **Run metadata**: Start/end times, duration, status

### Nested Runs for Tuning

When running `scripts/tune.py`:

1. A parent run is created with the study name (e.g., `crowd_counting_baseline`)
2. Each trial in the optimization creates a nested child run
3. Each trial executes a full training loop with different hyperparameters
4. MLflow tracks:
   - Which hyperparameters were tried in each trial
   - Which metrics improved or degraded
   - Which trial produced the best model
5. The best model is automatically registered to the MLflow model registry

**Example structure**:
```
Parent Run: crowd_counting_baseline (OptunaTuner)
  ├─ Trial 1: tune_attempt_001 (train with config A)
  ├─ Trial 2: tune_attempt_002 (train with config B)
  └─ Trial N: tune_attempt_NNN (train with config N)
```

## Tracking Key Configuration Changes

### Data Preprocessing

If dataset processing changed (e.g., new cleanup, different normalization, label scaling adjustments), check:

- **Loss function**: Inspect `loss_function` param in the run to see if mask-aware or MSE-only losses were used
- **Label scaling**: The `LABEL_SCALER` constant (currently 1000) affects density map magnitudes
- **Augmentation**: `aug_factor` and `num_ops` control photometric augmentation intensity
- **Crop size**: `crop_size` affects spatial context during training

### Model Architecture Changes

Configuration changes that affect the model:

- **Backbone**: `backbone` (e.g., `tu-convnext_base`) and `backbone_weights` source
- **Decoder channels**: `decoder_out_channels` affects feature richness
- **Attention mechanism**: `decoder_attention_type` can add spatial gating (optional)
- **Dropout**: `dropout` controls regularization strength
- **Trainability**: `trainable_backbone` determines if encoder weights are frozen or fine-tuned

### Loss Function Evolution

Historical loss choices:

- **`mask_mse_ssim`**: Composite loss combining density MSE, SSIM perceptual loss, and a learnable foreground mask
- Runs using `mask_mse_ssim` have additional metrics: `mask_iou`, `mask_dice` (segmentation performance)

### Learning Rate and Scheduling

Training dynamics tracked via:

- **`lr`**: Base learning rate
- **`lr_schedule`**: Schedule type (e.g., `clipped_exp` for exponential decay)
- **`scheduler_kwargs`**: Schedule-specific parameters (e.g., `decay_rate`, `min_lr_pct`)

## Finding Relevant Historical Runs

### Filter by Configuration

Use MLflow's run filter to find runs with specific properties:

```python
import mlflow
mlflow.set_experiment("crowd_counting")

# Find runs with a specific backbone
runs = mlflow.search_runs(
    filter_string="params.backbone = 'tu-convnext_base'",
    order_by=["metrics.val_mae ASC"],  # Sort by best validation MAE
    max_results=10
)
```

### Key Metrics to Compare

When evaluating historical runs:

- **`val_mae`** - Mean Absolute Error on validation set (primary metric)
- **`val_rmse`** - Root Mean Squared Error (penalizes outliers more)
- **`val_nae`** - Normalized Absolute Error (scale-independent)
- **`val_mbe`** - Mean Bias Error (systematic over/underestimation)
- **`best_val_mae`**, **`best_val_rmse`**, etc. - Best values achieved during training

## Checkpoint and Registry Relationship

1. **During training**: The best checkpoint (by `monitor_metric`) is saved locally to `./checkpoints/`
2. **After training**: The checkpoint weights and config are extracted and logged to MLflow
3. **Registration**: The run info is registered to the MLflow model registry (version 1, 2, 3, ...)
4. **Production**: A specific version can be promoted to a production alias for serving

This ensures that:
- Every checkpoint is tied to its exact training configuration
- Historical runs can be reproduced by loading the same checkpoint + config
- Model versions are comparable side-by-side

## Debugging Training Issues

When a training run fails or produces unexpected results:

1. **Check params** - Verify hyperparameters match intent (e.g., learning rate, batch size)
2. **Compare metrics** - Look at validation loss curves; identify sudden divergences
3. **Review data** - Check if data preprocessing or augmentation was misconfigured
4. **Inspect artifacts** - Download the checkpoint and configuration to reproduce locally
5. **Cross-reference branches** - If using git, identify which code version was used

Most of this metadata is centralized in MLflow runs, making historical analysis tractable.
