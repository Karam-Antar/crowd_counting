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


class BasePadCollate:
    """Base class to share mathematical padding logic across collate functions."""
    def __init__(self, multiple):
        self.multiple = multiple
        
    def get_padded_dims(self, h, w):
        """Calculates the nearest multiple of stride for height and width."""
        pad_h = ((h + self.multiple - 1) // self.multiple) * self.multiple
        pad_w = ((w + self.multiple - 1) // self.multiple) * self.multiple
        return pad_h, pad_w
        
    def pad_tensor(self, tensor, target_h, target_w):
        """Pads a tensor on the bottom and right edges using PyTorch's native F.pad."""
        h, w = tensor.shape[-2], tensor.shape[-1]
        pad_bottom = target_h - h
        pad_right = target_w - w
        
        # Skip padding entirely if already at target dimensions
        if pad_bottom == 0 and pad_right == 0:
            return tensor
            
        # F.pad expects padding for the last 2 dims formatted as: 
        # (pad_left, pad_right, pad_top, pad_bottom)
        return F.pad(tensor, (0, 0, pad_right, pad_bottom))


class DynamicPadCollate(BasePadCollate):
    def __call__(self, batch):
        # 1. Get original sizes (H, W are always the last two dimensions)
        original_sizes = [[item[0].shape[-2], item[0].shape[-1]] for item in batch]
        
        # 2. Find max H and max W across the batch
        max_h = max(size[0] for size in original_sizes)
        max_w = max(size[1] for size in original_sizes)
        
        # 3. Calculate the target padded dimensions for this batch
        pad_h, pad_w = self.get_padded_dims(max_h, max_w)
        
        # 4. Pad each item and stack them (F.pad natively fills with 0)
        batched_images = torch.stack([self.pad_tensor(item[0], pad_h, pad_w) for item in batch])
        batched_masks = torch.stack([self.pad_tensor(item[1], pad_h, pad_w) for item in batch])
        
        return batched_images, batched_masks, original_sizes


class FiveCropCollate(BasePadCollate):
    def __call__(self, batch):
        # 1. Stack and flatten the batch of crops
        imgs = torch.stack([item[0] for item in batch]).flatten(0, 1)
        masks = torch.stack([item[1] for item in batch]).flatten(0, 1)
        
        # 2. Get original unpadded size (all crops naturally share the same size)
        orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
        
        # 3. Calculate target padded dimensions
        pad_h, pad_w = self.get_padded_dims(orig_h, orig_w)
        
        # 4. Apply padding (Vectorized over the entire batch at once)
        padded_imgs = self.pad_tensor(imgs, pad_h, pad_w)
        padded_masks = self.pad_tensor(masks, pad_h, pad_w)
        
        # 5. Format original sizes to match the LitModel expectations
        B_total = imgs.shape[0]
        original_sizes = [[orig_h, orig_w]] * B_total
        
        return padded_imgs, padded_masks, original_sizes