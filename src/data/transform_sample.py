import timm
import torch
from torchvision.transforms import v2

from src.data.transform import PadToMultiple


def preprocess(img, params):
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