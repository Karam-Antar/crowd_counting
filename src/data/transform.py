from torchvision.transforms import v2
import torch
from torchvision.transforms.v2 import functional as F

class PadToMultiple(torch.nn.Module):
    """Pads image and mask to the nearest multiple of 'multiple'."""
    def __init__(self, multiple=32):
        super().__init__()
        self.multiple = multiple

    def forward(self, *inputs):
        # We assume the first input is the image/tensor to get dimensions from
        h, w = inputs[0].shape[-2:]
        
        ph = (self.multiple - h % self.multiple) % self.multiple
        pw = (self.multiple - w % self.multiple) % self.multiple
        
        # Padding format: (left, top, right, bottom)
        # We pad only the right and bottom to keep coordinates simple
        padding = [0, 0, pw, ph] 
        
        # Apply the same padding to image and mask
        return [F.pad(item, padding, fill=0) for item in inputs]
    



class CustomRandomCrop:
    """
    Handles padding if the image is smaller than patch_size, 
    then applies an identical random crop to both image and mask.
    """
    def __init__(self, patch_size=256):
        self.patch_size = patch_size

    def __call__(self, img, mask):
        # img and mask are expected to be PIL Images or Tensors
        h, w = F.get_size(img)
        
        # 1. Dynamic Padding (Safety Net)
        if w < self.patch_size or h < self.patch_size:
            pad_w = max(0, self.patch_size - w)
            pad_h = max(0, self.patch_size - h)
            
            # RGB image gets reflection padding to stay natural
            img = F.pad(img, (0, 0, pad_w, pad_h), padding_mode='constant', fill=0)
            # Density map MUST get zero padding to avoid 'ghost' people
            mask = F.pad(mask, (0, 0, pad_w, pad_h), fill=0)

        # 2. Get random crop coordinates
        # We calculate once and apply twice to ensure alignment
        i, j, h, w = v2.RandomCrop.get_params(
            img, output_size=(self.patch_size, self.patch_size)
        )
        
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        
        return img, mask