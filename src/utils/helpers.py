"""General utility functions that don't fit into specialized modules."""
from collections import defaultdict
import datetime
import glob
import multiprocessing
from pathlib import Path
import re
from PIL import Image
import h5py
import litlogger
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import optuna
from src import config

def get_study_best_value(trial: optuna.Trial):
    """Get the best validation accuracy from the study."""
    best_value = -float("inf")
    try:
        best_value = trial.study.user_attrs.get("val_accuracy", -float("inf"))
    except:
        pass
    return best_value


def forward_pass_batch(model, images, device=config.device):
    """Run a batch of images through a model and get logits.
    
    Centralizes the forward pass logic for both PyTorch and Lightning models.
    Handles device movement, eval mode, and gradient disabling.
    
    Args:
        model: PyTorch or Lightning model.
        images: Batch of input images (B, C, H, W).
        device: Device to run inference on.
        
    Returns:
        Logits tensor of shape (B, num_classes).
    """
    model.eval()
    model = model.to(device)
    images = images.to(device)
    
    with torch.no_grad():
        logits = model(images)
    
    return logits


def dict_to_str(d: dict):
    return {str(k): str(v) for k, v in d.items()}


def flatten_dict(nested_dict, separator='.', prefix='', to_str=False):
    """
    Flattens a nested dictionary into a single-level dictionary.
    
    Args:
        nested_dict (dict): The dictionary to flatten.
        separator (str): Character to join keys (default is '.').
        prefix (str): Internal use for recursion.
        
    Returns:
        dict: A flat dictionary with path-based keys.
    """
    flat_dict = {}
    
    for key, value in nested_dict.items():
        # Create the new key path
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict) and value:
            # If the value is a non-empty dict, recurse deeper
            flat_dict.update(flatten_dict(value, separator, new_key))
        else:
            # If it's a leaf node (or empty dict), assign the value
            flat_dict[new_key] = value
    
    if to_str:
        flat_dict = dict_to_str(flat_dict)
            
    return flat_dict

def unflatten_dict(flat_dict, separator='.'):
    """
    Reconstructs a nested dictionary from a flattened one.
    
    Args:
        flat_dict (dict): The flat dictionary with path-based keys.
        separator (str): Character used to split keys.
        
    Returns:
        dict: The original nested dictionary structure.
    """
    nested_dict = {}
    
    for composite_key, value in flat_dict.items():
        parts = composite_key.split(separator)
        current_level = nested_dict
        
        # Iterate through the path parts, creating dicts as we go
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        
        # Set the value at the very last part of the path
        current_level[parts[-1]] = value
        
    return nested_dict


def get_unique_experiment_name(base_name):
    """Generate a unique experiment name by appending a compact timestamp."""
    
    
    # Already exists — append compact timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # e.g., "0507-1432"
    return f"{base_name}_{timestamp}"


def create_empty_text_files(target_path: str, file_names: list[str]):
    """
    Creates empty .txt files for each string in a list at a specific path.
    
    Args:
        file_names (list): List of strings to be used as filenames.
        target_path (str): The directory where files should be created.
    """
    # 1. Convert the string path to a Path object and create it if it doesn't exist
    directory = Path(target_path)
    directory.mkdir(parents=True, exist_ok=True)
    
    for name in file_names:
        # 2. Ensure the filename ends with .txt
        if not name.endswith('.txt'):
            filename = f"{name}.txt"
        else:
            filename = name
            
        # 3. Join the directory with the filename
        file_path = directory / filename
        
        # 4. Create the empty file
        file_path.touch()
        # print(f"Created: {file_path}")

def generate_file_names(metrics: dict):
    # 1. Group metrics by their phase ('val' or 'train')
    grouped_metrics = defaultdict(dict)
    
    for k, v in metrics.items():
        # Example k: "best_val_MAE" -> splits into ["best", "val", "MAE"]
        parts = k.split('_', 2) 
        
        if len(parts) >= 3 and parts[0] == "best":
            phase = parts[1]         # 'val' or 'train'
            metric_name = parts[2]   # 'loss', 'MAE', 'MSE', etc.
            
            grouped_metrics[phase][metric_name] = v

    # 2. Build the formatted strings
    names = set()
    for phase, phase_metrics in grouped_metrics.items():
        # Dynamically build the inner string: e.g., "(loss=0.0123)(MAE=4.5678)"
        stats_str = "".join([f"({m_name}={float(m_val):.4f})" for m_name, m_val in phase_metrics.items()])
        
        # Combine phase and stats: e.g., "val(loss=0.0123)(MAE=4.5678)"
        names.add(f"{phase}{stats_str}")

    return list(names)


def to_snake_case(name: str) -> str:
    """
    Converts PascalCase or camelCase to snake_case.
    Example: 'TransferModel' -> 'transfer_model'
    """
    # 1. Insert an underscore before any capital letter followed by a lowercase letter
    # 2. Insert an underscore between a lowercase and uppercase letter
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    
    # 3. Handle cases with consecutive capitals (like 'CNNModel' -> 'cnn_model')
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    
    return s2.lower()


def check_experiment_existence(name: str, queue: multiprocessing.Queue):
    try:
        # Ping the cloud API safely inside the sandbox
        exp = litlogger.init(name=name)
        has_data = len(exp.metadata) > 0
        
        # If it's taken, cleanly close the scout connection
        # if has_data:
        exp.finalize('aborted')
            
        # Send the boolean answer back to the main process
        queue.put(has_data)
    except Exception as e:
        print(f"Cloud scout warning: {e}")
        queue.put(False) # Safe fallback if API errors out


def get_sample_from_dm(datamodule, index=0):
    sample_image, _ = datamodule.val_ds[index]
    sample_image = sample_image.unsqueeze(0)
    return sample_image

def get_sample_from_ds(img_path=None, h5_path=None, index=0):
    # Load Image
    if not img_path:
        img_path = glob.glob(f"{config.DATASET_PATH}/train_data/images/*.jpg")[index]
        h5_path = img_path.replace("images", "ground-truth-h5").replace(".jpg", ".h5")
    img = Image.open(img_path).convert('RGB')
    
    # Load Density Map
    with h5py.File(h5_path, 'r') as hf:
        density_map = np.asarray(hf['density'])
    
    # Calculate Ground Truth Count
    gt_count = np.sum(density_map)
    return img, density_map, gt_count