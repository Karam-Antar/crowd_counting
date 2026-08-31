<!-- # Model Artifact Overview

Model artifacts are the files and metadata generated during training that capture the trained model state, configuration, and performance metrics. Understanding which artifacts are active, historical, or intermediate helps with reproducibility and debugging.

## Artifact Locations and Types

### Local Checkpoints

**Location**: `./checkpoints/` (created during training)

**Files**:
- **`checkpoint.ckpt`** - PyTorch Lightning checkpoint (binary format)
  - Contains: model weights, training state, optimizer state
  - Used for: resuming training, local evaluation, debugging

**Lifecycle**:
- Created during each training run
- Only the best checkpoint (by `monitor_metric`) is saved
- Overwritten on the next training run (if same directory is used)
- Should be archived or renamed if results are valuable

### Model Configuration Files

**Filename**: `model_config.json`

**Location**: 
- Stored in MLflow artifact repository alongside the model
- Also saved locally during `StandardRunner` execution

**Contents**: Serialized `BaseParams` dataclass including:
```json
{
  "model_class": "MAnet",
  "backbone": "tu-convnext_base",
  "crop_size": 480,
  "batch_size": 8,
  "loss_function": "mask_mse_ssim",
  "lr": 0.00065,
  "epochs": 60,
  "stop_patience": 35
}
```

**Lifecycle**:
- Created at the start of each training run
- Uploaded to MLflow alongside the checkpoint
- Serves as the single source of truth for reproduction

### PyTorch State Dictionary

**Filename**: `weights.pt`

**Location**: 
- Extracted from checkpoint during MLflow registration
- Stored in MLflow artifacts

**Contents**:
- Pure model weights (no optimizer or trainer state)
- Maps layer names to parameter tensors
- Lightweight compared to full checkpoint

**Usage**:
```python
import torch
from src.models.model import CrowdCounter
from src.core.params import BaseParams

params = BaseParams.from_json('model_config.json')
model = CrowdCounter(params=params)
state_dict = torch.load('weights.pt', map_location='cpu')
model.load_state_dict(state_dict)
```

## MLflow Registry Artifacts

### Registered Models

When a model is registered to MLflow via `MLFlowRegistry.upload_model()`:

**Model URI**: `models:/crowd_counting/VERSION`

**Artifacts stored**:
1. **pyfunc model** - The `ProductionPyTorchWrapper` class that serves predictions
2. **weights.pt** - State dictionary extracted from the checkpoint
3. **model_config.json** - Configuration snapshot from training
4. **code_paths** - Complete `src/` directory for runtime imports
5. **metadata directory** - Additional metadata and experiment tracking files

**Versions**:
- Version 1, 2, 3, ... incremented with each registration
- Can be aliased (e.g., `production`, `staging`, `latest`) for easier reference

### Model Metadata Artifacts

**Stored in MLflow under**: `{model_name}/`

**Typical files**:
- **`config_reference.json`** - Full hyperparameter snapshot
- **`training_results.json`** - Final validation and test metrics
- **`experiment_info.txt`** - Run ID, timestamp, data version
- **Source code archive** - Zipped `src/` directory for full reproducibility

## Run-Level Artifacts

### Training Logs

**Location**: MLflow run artifacts under each training run

**Contents**:
- Epoch-by-epoch metrics (loss, validation metrics)
- Learning rate schedule tracking
- Sample outputs (if enabled)

### Experiment Metadata

**Stored in MLflow run**:

- **`params/`** - Hyperparameters as key-value pairs
- **`metrics/`** - Metric history (one entry per epoch)
- **`tags/`** - Custom metadata (run purpose, notes, etc.)
- **`artifacts/`** - Checkpoint, config, and any custom outputs

## Interpreting Artifact Status

### Active Artifacts

- **In-use models**: Versions in MLflow registry with an alias (e.g., `production`, `staging`)
- **Recent checkpoints**: Created in the last few weeks during active development
- **Current config**: The `model_config.json` in the latest run

**How to identify**: Check MLflow UI for recent runs with high metric values

### Historical Artifacts

- **Previous model versions**: Earlier registered versions that were superseded
- **Archived checkpoints**: Older local checkpoints no longer needed
- **Experiment trials**: Individual runs from hyperparameter tuning

**How to identify**: Sort MLflow runs by date; older runs are historical

### Intermediate/Temporary Artifacts

- **Local `./checkpoints/`** - Overwritten by each new training run
- **Tuning trial checkpoints** - Generated during hyperparameter search, not registered
- **Logs and temporary files** - Training progress files that aren't needed after completion

**How to identify**: Short-lived files outside MLflow; not part of the final model package

## Reproducibility Contract

To fully reproduce a historical training run:

1. **Load the checkpoint** → `weights.pt` and Lightning state
2. **Load the config** → `model_config.json` to recreate the exact architecture
3. **Load the data** → Use the dataset split noted in the run metadata
4. **Load the code** → Use the source archive or git commit ID recorded in MLflow

This ensures complete reproducibility without ambiguity about which version of the code or config was used.

## Cleanup and Archival

**What to keep**:
- All MLflow runs (these are the source of truth)
- Registered model versions (immutable and versioned)
- Config files from important runs (lightweight, high information value)

**What can be deleted**:
- Local `./checkpoints/` after uploading to MLflow
- Tuning trial checkpoints after the best model is selected
- Temporary experiment files outside MLflow

**Best practice**:
- Archive important local checkpoints with a descriptive name before overwriting
- Let MLflow handle long-term storage of production models
- Use git to version code alongside MLflow artifact versioning -->
