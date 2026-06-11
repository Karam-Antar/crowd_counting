"""Model persistence functions for saving, loading, and logging models."""

from datetime import datetime
import os
import torch
import lightning.pytorch as pl
from torch.export import export, save, load, Dim
from src import config
from src.core.params import BaseParams


def prepare_model_to_export(params, model):
    image_size = int(params.image_size) if params.image_size else 256
    example_input = torch.randn(4, 3, image_size, image_size)
    batch = Dim("batch", min=1, max=1024)
    height = Dim("height", min=1, max=1024) 
    width = Dim("width", min=1, max=1024)
    model.eval()
    dynamic_shapes = {
            "x": {
                0: batch,   # dynamic batch size
                # 1: Dim('channels'),
                2: params.padding_multiple * height,  # dynamic height
                3: params.padding_multiple * width,   # dynamic width
            }
        }
    return example_input, dynamic_shapes

def save_model(model: pl.LightningModule, params: BaseParams, architecture: str = '', model_id: str = None, model_path: str | None = None, extension: str = 'pt2'):
    """Save a PyTorch Lightning model to disk in TorchScript format.

    Args:
        model: PyTorch model to save.
        architecture: Name of the architecture.
        model_id: ID for the saved model (uses timestamp if not provided).
        extension: File extension (default 'pt' for TorchScript).
    """
    model_path = model_path if model_path else os.path.join(config.MODEL_SAVE_PATH, architecture)
    os.makedirs(model_path, exist_ok=True)

    if model_id is None:
        model_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(model_path, f'{model_id}.{extension}')

    # Convert the LightningModule to TorchScript and save it directly.
    # By default, Lightning only scripts the forward() pass for inference.
    # model.to_torchscript(file_path=model_path, method='script')
    example_input, dynamic_shapes = prepare_model_to_export(params, model)
    exported_model = export(
        model,
        (example_input,),
        dynamic_shapes=dynamic_shapes
    )
    save(exported_model, model_path)

    print(f"Model saved in export format to {model_path}")
    return exported_model, model_path


def load_model(model_path: str, device: str = config.device):
    """Load a saved TorchScript model from disk.

    Args:
        model_path: Path to the saved scripted model.
        device: Device to load model on ('cpu', 'cuda', etc.).
    """
    # Load the TorchScript model
    # torch.jit.load automatically restores the graph and the weights
    model = load(model_path)
    # model.eval()

    print(f"Model loaded from {model_path}")
    return model
