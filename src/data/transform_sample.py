import timm
import torch
from torchvision.transforms import v2

from src.data.transform import PadToMultiple


def preprocess(img, params):
    """Normalize a raw image tensor using the model's configured preprocessing pipeline.

    Args:
        img: Input image tensor or batch of tensors with shape ``(B, H, W, C)`` or
            ``(H, W, C)``.
        params (BaseParams): Experiment configuration containing the backbone and
            padding parameters needed for normalization.

    Returns:
        torch.Tensor | list[torch.Tensor]: The preprocessed image tensor or list of
            tensors matching the model's expected input layout.

    Raises:
        ValueError: If the input tensor cannot be converted into a valid image format.
    """
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
    return img[0] if isinstance(img, list) and len(img) == 1 else img