# Serving

Serving for this project uses MLflow's PyFunc interface to wrap the trained CrowdCounter model. The serving pipeline maintains strict consistency with the training pipeline to ensure reliable inference.

## MLflow PyFunc Wrapper

The model is served through `ProductionPyTorchWrapper`, a custom MLflow PythonModel that:

1. **Loads at startup**: Resolves model weights and configuration from MLflow artifacts
2. **Restores state**: Maps the checkpoint to the correct device (CUDA or CPU)
3. **Exposes predict method**: Accepts NumPy arrays and returns structured predictions

The wrapper is defined in `src/core/pyfunc.py` and handles:

- Device detection (GPU if available, falls back to CPU)
- Model initialization with the exact training hyperparameters
- Inference-mode locking (`.eval()` to disable dropout and batch norm)

## Required Artifacts

A production model requires three artifacts in MLflow:

1. **weights** - PyTorch state dictionary (`weights.pt`) containing trained model parameters
2. **params** - JSON configuration file (`model_config.json`) with the complete hyperparameter set
3. **code_paths** - The full `src/` directory for runtime imports and model class definitions

## Expected Input Format

Serving expects **RGB images** as NumPy arrays with shape `(H, W, 3)` and dtype `uint8`:

```python
import numpy as np
from PIL import Image

# Load an image
img = Image.open('path/to/image.jpg')
img_array = np.array(img)  # Shape: (H, W, 3), dtype: uint8

# Serve through MLflow
output = model.predict(context, img_array)
```

### Input Preprocessing

The inference pipeline normalizes inputs to match the training normalization:

- **Channel order**: RGB (PIL and OpenCV default; not BGR)
- **Value range**: uint8 (0-255) is normalized to float32 (0-1) internally
- **Spatial dimensions**: No fixed size required; the model adapts to arbitrary input shapes via padding

## Expected Output Format

The model returns a dictionary with:

- **"count"** - Scalar float representing the total estimated crowd count
- **"density_map"** - 2D float array matching input spatial dimensions, where each pixel value represents local crowd density

```python
output = {
    "count": 342.5,  # Total crowd estimate
    "density_map": array([[0.01, 0.05, ...], [0.03, 0.08, ...], ...])  # Per-pixel density
}
```

The density map is scaled back to the original image size automatically during inference preprocessing, and the count is computed as the sum of the density map.

## Consistency with Training

Critical assumptions that link serving to training:

1. **Normalization**: Input images are normalized using the same `ImageNet` statistics used during training
2. **Padding strategy**: Spatial dimensions are padded to multiples of 32 (required by the encoder-decoder backbone)
3. **Label scaling**: Density outputs use `LABEL_SCALER = 1000` to match the scaling applied during training
4. **Model state**: Inference automatically disables dropout and uses batch norm in evaluation mode

If the training pipeline changes (e.g., new normalization, different backbone, altered preprocessing), the serving pipeline must be updated and a new model version registered.

## Loading and Serving Models

### From MLflow Registry

```python
import mlflow

# Load a specific model version
model_uri = "models:/crowd_counting/production"  # or "models:/crowd_counting/1"
model = mlflow.pyfunc.load_model(model_uri)

# Predict on images
result = model.predict(image_array)
```

### From Local Checkpoint

```python
from src.core.model_wrapper import ModelWrapper
from src.models.model import CrowdCounter
from src.core import inference

# Wrap a local checkpoint
model = CrowdCounter(params=params)
model.load_state_dict(torch.load('checkpoint.pt', map_location='cpu'))
wrapper = ModelWrapper(model)

# Predict
count, density_map = wrapper.predict(image_tensor)
```
