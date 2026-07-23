from torchvision.transforms import v2
import torch
from torchvision.transforms.v2 import functional as F
import random

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
    
class UnpadToOriginal(torch.nn.Module):
    """Removes the right and bottom padding to restore the tensor to its original dimensions."""
    def __init__(self):
        super().__init__()

    def forward(self, padded_tensor, original_shape):
        """
        Args:
            padded_tensor (Tensor): The model's output tensor (e.g., shape [B, C, H_padded, W_padded])
            original_shape (tuple or torch.Size): The (height, width) of the image before padding
        """
        # Extract the original height and width
        org_h, org_w = original_shape[-2:]
        
        # Use ellipsis (...) to cleanly handle any batch or channel dimensions
        return padded_tensor[..., :org_h, :org_w]


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



class SafePhotometricRandAugment(torch.nn.Module):
    """
    Mimics v2.RandAugment but restricts operations strictly to pixel-level (color/lighting) 
    transforms. This prevents geometric operations from misaligning the image and density map.
    """
    def __init__(self, num_ops: int = 2, magnitude: int = 0.3):
        super().__init__()
        self.num_ops = num_ops
        
        # Scale magnitude safely to a [0, 1] range factor
        mag_scale = magnitude
        
        # Instantiate safe, linear operations ONCE during initialization.
        # Notice internal probabilities are set to 1.0 because execution 
        # randomness is controlled by our random.sample selection process.
        self.op_pool = torch.nn.ModuleList([
            v2.ColorJitter(
                brightness=0.6 * mag_scale, 
                contrast=0.6 * mag_scale, 
                saturation=0.6 * mag_scale, 
                hue=0.08 * mag_scale
            ),
            v2.RandomGrayscale(p=mag_scale),
            # Kernel size must be an odd integer
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.1 + 1.5 * mag_scale)),
            v2.RandomAdjustSharpness(sharpness_factor=1.0 + mag_scale, p=1.0)
        ])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # 1. Pre-clamp guard: Ensure input is clean [0, 1] float data
        img = torch.clamp(img, 0.0, 1.0)
        
        # 2. Randomly sample N distinct operations from our safe pool
        num_to_sample = min(self.num_ops, len(self.op_pool))
        sampled_indices = random.sample(range(len(self.op_pool)), num_to_sample)
        
        # 3. Apply operations sequentially
        for idx in sampled_indices:
            img = self.op_pool[idx](img)
            
        # 4. Post-clamp guard: Prevent any pixel overflow from reaching the network
        return torch.clamp(img, 1e-6, 1.0)


class DynamicPadCollate:
    def __init__(self, multiple):
        self.multiple = multiple
        
    def __call__(self, batch):
        B = len(batch)
        C = batch[0][0].shape[0]  
        stride = self.multiple
        
        # 1. Get original sizes
        original_sizes = torch.tensor([(item[0].shape[1], item[0].shape[2]) for item in batch])
        
        # 2. Find max H and max W across the batch
        max_h, max_w = original_sizes.max(dim=0).values.tolist()
        
        # 3. Adjust max_h and max_w to be multiples of the stride (32)
        # Math trick to round up to the nearest multiple of 'stride'
        pad_h = ((max_h + stride - 1) // stride) * stride
        pad_w = ((max_w + stride - 1) // stride) * stride
        
        # 4. Pre-allocate tensors using the stride-padded dimensions
        batched_images = torch.zeros((B, C, pad_h, pad_w), dtype=batch[0][0].dtype)
        
        mask_shape = batch[0][1].shape
        if len(mask_shape) == 3:  
            batched_masks = torch.zeros((B, mask_shape[0], pad_h, pad_w), dtype=batch[0][1].dtype)
        else:                     
            batched_masks = torch.zeros((B, pad_h, pad_w), dtype=batch[0][1].dtype)

        # 5. Drop images into the top-left corner
        for i, (img, mask) in enumerate(batch):
            h, w = original_sizes[i].tolist()
            
            batched_images[i, :, :h, :w] = img
            
            if len(mask_shape) == 3:
                batched_masks[i, :, :h, :w] = mask
            else:
                batched_masks[i, :h, :w] = mask
                
        return batched_images, batched_masks, original_sizes.tolist()