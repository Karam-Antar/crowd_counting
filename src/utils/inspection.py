"""Model inspection utilities for examining model architectures."""

import timm
from torchvision.models.feature_extraction import get_graph_node_names 


def inspect_model(model_name):
    """Utility to print out the architecture of a timm model."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    for name, module in model.named_modules():
        if name: # Avoid printing the root (empty string)
            print(f"{name}: {type(module).__name__}")


def graph_node_names(model, layer_name=None):

    # Get all available nodes in the model
    train_nodes, eval_nodes = get_graph_node_names(model)

    # This will print every single string you can use in return_nodes
    for node in eval_nodes:
        if not layer_name:
            print(node)
            continue
        if layer_name in node: # Filter to find conv layers specifically
            print(node)