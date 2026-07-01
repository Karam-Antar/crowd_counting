"""Visualization functions for model predictions, activations, and performance metrics.

This module provides visualization utilities for PyTorch Lightning models and native PyTorch models.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import lightning.pytorch as pl
import torchlens
from src import config
from .helpers import forward_pass_batch
from torchvision.transforms import v2
from torchvision.models.feature_extraction import create_feature_extractor
from src.utils import helpers
import h5py
from PIL import Image
import glob


def _prepare_image_for_display(image_tensor):
    """
    Prepares a single image tensor for display with matplotlib.
    Handles channel permutation (C, H, W -> H, W, C) and converts to numpy.
    """
    # Ensure it's on CPU and convert to numpy
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = v2.ToImage()(image_tensor)
    img_np = image_tensor.cpu().numpy()

    # Handle batch dimension if present (take the first image)
    if len(img_np.shape) == 4:
        img_np = img_np[0]

    # Transpose if channels are first (C, H, W -> H, W, C)
    if img_np.shape[0] in [1, 3] and img_np.ndim == 3: # 1 channel (grayscale) or 3 channels (RGB)
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Remove single channel dimension for grayscale images if present (H, W, 1 -> H, W)
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        img_np = img_np.squeeze(2)

    return img_np



def display_original_image(input_image):
    """Display the original input image.
    
    Args:
        input_image: Input tensor of shape (H, W, C) or (B, H, W, C).
    """
    display_img = _prepare_image_for_display(input_image) 
    plt.imshow(display_img, aspect='auto')
    plt.axis('off')
    plt.show() 



def visualize_sample(img: torch.Tensor, target: tuple | None = None, pred: tuple | None = None, cmap='jet'):
    if target is not None:
        target_count, target_map = target
    if pred is not None:
        pred_count, pred_map = pred
    # Create a 1x3 grid of subplots
    cols = 1 + (target is not None) + (pred is not None)
    fig, ax = plt.subplots(1, cols, figsize=(18, 5))
    img = _prepare_image_for_display(img)
    if target is not None:
        target_map = _prepare_image_for_display(target_map)
    if pred is not None:
        pred_map = _prepare_image_for_display(pred_map)
    # 1. Original Image
    index = 0
    ax[index].imshow(img)
    ax[index].set_title("Original Image")
    ax[index].axis('off')
    index += 1
    
    if target is not None:
    # 2. Ground Truth Density Map
        ax[index].imshow(target_map, cmap=cmap)
        ax[index].set_title(f"Ground Truth (Count: {target_count:.2f})")
        ax[index].axis('off')
        index += 1
    
    # 3. Predicted Density Map
    if pred is not None:
        ax[index].imshow(pred_map, cmap=cmap)
        ax[index].set_title(f"Prediction (Count: {pred_count:.2f})")
        ax[index].axis('off')
    
    plt.tight_layout()
    plt.show()




def preprocess_image(input_tensor):
    """Ensure the tensor has a batch dimension."""
    if len(input_tensor.shape) == 3:  # Single image (C, H, W)
        input_tensor = input_tensor.unsqueeze(0)
    return input_tensor.float()

# def display_original_image_pt(input_tensor):
#     """Display the original PyTorch tensor image."""
#     # Grab first image in batch and convert from (C, H, W) to (H, W, C) for matplotlib
#     display_img = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    
#     # Normalize to [0, 1] just in case the tensor is normalized with mean/std
#     display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)
    
#     plt.imshow(display_img, aspect='auto')
#     plt.axis('off')
#     plt.show()

# --- Step 2: Extract outputs from the requested layers ---
def get_layer_output(model, img_tensor, layer_name):
    """Build a sub-model up to the target layer and run prediction (Keras style)."""
    model.eval()
    print('image shape: ', img_tensor.shape)
    # This creates a model that outputs a dict of requested layers.
    activation_model = create_feature_extractor(model, return_nodes={layer_name: 'output'})
    
    with torch.no_grad():
        # We perform the forward pass through our new extractor
        outputs = activation_model(img_tensor)
        
    layer_output = outputs['output']
    return layer_output


def visualize_filter(channel_image, filter_index, layer_name, cmap):

    # Normalize for better visualization (matching your exact logic)
    p_low, p_high = np.percentile(channel_image, (1, 99))
    channel_image = np.clip(channel_image, p_low, p_high)
    channel_image -= channel_image.min()
    channel_image /= (channel_image.max() + 1e-8)

    # Plot the single selected filter channel
    plt.figure(figsize=(6, 6)) # Square sizing looks great for individual maps
    plt.title(f"{layer_name} - Filter Index {filter_index}")
    plt.grid(False)
    plt.axis('off') 
    plt.imshow(channel_image, cmap=cmap)
    plt.show()

# --- Step 3: Visualize feature maps ---
def visualize_feature_maps_grid(layer_output, layer_name, cmap, filter_index):
    """Tile feature maps into a single grid array and display them with one imshow."""
    # PyTorch outputs (B, C, H, W). We take the first item in the batch.
    features = layer_output[0].detach().numpy()  # Shape: (C, H, W)
    n_features = features.shape[0]
    h, w = features.shape[1], features.shape[2]

    if filter_index is not None:
        # Safety Check
        if filter_index < 0 or filter_index >= n_features:
            raise ValueError(f"Requested filter_index {filter_index}, but layer '{layer_name}' only has {n_features} channels.")
        
        # Pull single feature map
        channel_image = features[filter_index, :, :]
        visualize_filter(channel_image, filter_index, layer_name, cmap)

    # Calculate grid dimensions (Max 8 columns to match your Keras styling)
    n_cols = min(8, n_features)
    n_rows = math.ceil(n_features / n_cols)
    
    # Create the single blank canvas
    display_grid = np.zeros((h * n_rows, w * n_cols))

    for idx in range(n_features):
        row = idx // n_cols
        col = idx % n_cols
        
        # Get the specific feature map
        channel_image = features[idx, :, :]

        # Normalize for better visualization (matching your exact Keras logic)
        p_low, p_high = np.percentile(channel_image, (1, 99))
        channel_image = np.clip(channel_image, p_low, p_high)
        channel_image -= channel_image.min()
        channel_image /= (channel_image.max() + 1e-8)

        # Place the feature map into the correct location on the grid
        display_grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = channel_image

    # Plot the final combined grid
    scale = 1.0 / max(h, w)
    # Adjust multiplier to make the figure a good size on screen
    plt.figure(figsize=(max(8, scale * display_grid.shape[1] * 20), 
                        max(8, scale * display_grid.shape[0] * 20)))
    plt.title(f"{layer_name} - Feature Maps")
    plt.grid(False)
    plt.axis('off') # Hiding axes for a cleaner look
    plt.imshow(display_grid, aspect='auto', cmap=cmap)
    plt.show()

# --- Main function that ties everything together ---
def visualize_feature_maps(model, history, layer_name, filter_index=None, input_tensor=None, cmap='gray', device=config.device):
    """
    Visualizes the feature maps of a specific PyTorch layer for a given input tensor.
    """
    model = model.to(device)
    if input_tensor is not None:
        display_original_image(input_tensor)
    # img_tensor = preprocess_image(input_tensor).to(device)
    
    # Extract and plot
    # layer_output = get_layer_output(model, img_tensor, layer_name)
    print(f"Visualizing layer: {layer_name}")
    visualize_feature_maps_grid(history[layer_name].activation, layer_name, cmap, filter_index)


def visualize_model_graph(model: torch.nn.Module, datamodule=None, sample=None):
    if sample is None:
        if datamodule is None:
            raise ValueError('Either sample or datamodule should be provided.')
        sample, _, _ = helpers.get_sample_from_dm(datamodule)
    # history = torchlens.log_forward_pass(model, helpers.get_sample_from_dm(datamodule))
    try: 
        graph = torchlens.visualization.show_model_graph(
            model, 
            sample, 
            vis_mode='unrolled',
            vis_direction='topdown',
        )
    except Exception as e:
        print(e)
        pass