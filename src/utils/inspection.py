"""Model inspection utilities for examining model architectures."""

import timm
from torchvision.models.feature_extraction import get_graph_node_names 


def inspect_model(model_name):
    """Print the module names for a pretrained timm model architecture.

    Args:
        model_name (str): Name of the timm model to instantiate.

    Returns:
        None: Model module names are printed to stdout.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    for name, module in model.named_modules():
        if name: # Avoid printing the root (empty string)
            print(f"{name}: {type(module).__name__}")


def graph_node_names(model, layer_name=None):
    """List graph node names for a model, optionally filtered by a layer substring.

    Args:
        model: PyTorch model instance.
        layer_name (Optional[str]): Optional substring filter to narrow graph nodes.

    Returns:
        None: Matching node names are printed to stdout.
    """
    # Get all available nodes in the model
    train_nodes, eval_nodes = get_graph_node_names(model)

    # This will print every single string you can use in return_nodes
    for node in eval_nodes:
        if not layer_name:
            print(node)
            continue
        if layer_name in node: # Filter to find conv layers specifically
            print(node)