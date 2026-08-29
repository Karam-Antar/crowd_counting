"""General utility functions that don't fit into specialized modules."""
from collections import defaultdict
import datetime
import multiprocessing
from pathlib import Path
import re
import litlogger
import torch
from src import config
from src.data.transform import PadToMultiple

def get_study_best_value(trial):
    """Return the best tracked validation value for a study if it exists.

    Args:
        trial: Optuna trial containing study-level user attributes.

    Returns:
        float: Best value discovered so far, or ``-inf`` when no value is recorded.
    """
    best_value = -float("inf")
    try:
        best_value = trial.study.user_attrs.get("val_accuracy", -float("inf"))
    except:
        pass
    return best_value


def forward_pass_batch(model, images, device=config.device):
    """Run a batch through a model while forcing evaluation mode and no gradients.

    Args:
        model: PyTorch or Lightning model instance.
        images (torch.Tensor): Batch of input images with shape ``(B, C, H, W)``.
        device: Target device for inference, such as ``"cuda"`` or ``"cpu"``.

    Returns:
        torch.Tensor: Model output for the provided image batch.
    """
    model.eval()
    model = model.to(device)
    images = images.to(device)
    
    with torch.no_grad():
        logits = model(images)
    
    return logits


def dict_to_str(d: dict):
    """Convert all dictionary keys and values into plain strings.

    Args:
        d (dict): Input dictionary.

    Returns:
        dict: String-keyed and string-valued dictionary snapshot.
    """
    return {str(k): str(v) for k, v in d.items()}


def flatten_dict(nested_dict, separator='.', prefix='', to_str=False):
    """Flatten a nested dictionary into a single-level representation.

    Args:
        nested_dict (dict): Dictionary to flatten.
        separator (str): Delimiter used between nested keys.
        prefix (str): Internal recursion prefix.
        to_str (bool): Whether to stringify keys and leaf values.

    Returns:
        dict: Flattened dictionary with composite keys.
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
    """Rebuild a nested dictionary structure from a flattened dictionary.

    Args:
        flat_dict (dict): Flat dictionary with dot-delimited keys.
        separator (str): Key separator used during flattening.

    Returns:
        dict: Nested dictionary reconstructed from the flattened form.
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
    """Create a unique experiment name based on the current timestamp.

    Args:
        base_name (str): Base experiment identifier.

    Returns:
        str: Experiment name suffixed with a compact timestamp.
    """
    # Already exists — append compact timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # e.g., "0507-1432"
    return f"{base_name}_{timestamp}"





def to_snake_case(name: str) -> str:
    """Convert CamelCase or PascalCase to snake_case.

    Args:
        name (str): Identifier to convert.

    Returns:
        str: Snake_case version of the provided identifier.
    """
    # 1. Insert an underscore before any capital letter followed by a lowercase letter
    # 2. Insert an underscore between a lowercase and uppercase letter
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    
    # 3. Handle cases with consecutive capitals (like 'CNNModel' -> 'cnn_model')
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    
    return s2.lower()


def check_experiment_existence(name: str, queue: multiprocessing.Queue):
    """Check whether an experiment name already exists in the tracking backend.

    Args:
        name (str): Experiment name to probe.
        queue (multiprocessing.Queue): Queue used to return the boolean result.

    Returns:
        None: The boolean status is emitted to the provided queue.
    """
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
    """Retrieve one sample from a datamodule's validation or test dataset.

    Args:
        datamodule: Data module exposing a ``test_ds`` or ``val_ds`` dataset.
        index (int): Index of the sample to retrieve.

    Returns:
        tuple[torch.Tensor, torch.Tensor, float]: Image tensor, target map, and GT count.
    """
    # Capture both the image and the target (density map) from the dataset
    sample_image, target = (datamodule.test_ds or datamodule.val_ds)[index]
    # Add batch dimension to the image
    sample_image = sample_image.unsqueeze(0)
    # Calculate the ground truth count by summing the density map tensor
    gt_count = target.sum().item()
    sample_image = PadToMultiple()(sample_image)[0]
    
    return sample_image, target, gt_count

def get_sample_from_ds(image_path, gt_dir, transform=None):
    """Load a single image and matching density-map target from disk.

    Args:
        image_path (str): Path to the RGB image.
        gt_dir (str): Directory or file path to the corresponding ground-truth density map.
        transform: Optional callable applied to the loaded image-target pair.

    Returns:
        tuple[torch.Tensor, torch.Tensor, float]: Loaded image, target tensor, and GT count.

    Raises:
        FileNotFoundError: If the expected target file path does not exist.
    """
    import os
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import tv_tensors
    # 1. Deduce the GT path from the image filename
    # base_name = os.path.basename(image_path)                  # e.g., "image_01.jpg"
    # file_name_without_ext = os.path.splitext(base_name)[0]    # e.g., "image_01"
    gt_path = gt_dir
    
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Expected ground truth file not found: {gt_path}")

    # 2. Load and convert Image
    with Image.open(image_path) as img:
        img_tensor = tv_tensors.Image(img.convert('RGB'))
    
    # 3. Load and convert Ground Truth (NPY)
    target_array = np.load(gt_path)
    target_tensor = tv_tensors.Mask(
        torch.from_numpy(target_array).float().unsqueeze(0)
    )
    gt_count = target_tensor.sum().item()

    # 4. Apply Transforms if provided
    if transform:
        img_tensor, target_tensor = transform(img_tensor, target_tensor)

    return img_tensor, target_tensor, gt_count

def transform_sample(img, params):
    """Apply the model's standard preprocessing transform to a single sample.

    Args:
        img: Single image or image batch in raw RGB format.
        params (BaseParams): Parameter object containing the backbone and padding settings.

    Returns:
        torch.Tensor: Preprocessed sample matching the model input format.
    """
    import timm
    from torchvision.transforms import v2
    data_config = timm.data.resolve_data_config({}, model=params.backbone)
    mean = data_config['mean']
    std = data_config['std']

    transforms = v2.Compose([
        v2.ToImage(),
        PadToMultiple(params.padding_multiple),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])
    img = transforms(img)
    return img[0]