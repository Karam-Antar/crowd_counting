from torchvision.transforms import v2
import torch
from torchvision.transforms.v2 import functional as F
import random

class PadToMultiple(torch.nn.Module):
    """Pad image and target tensors to the nearest multiple of a base stride.

    This is primarily used to align spatial dimensions with model downsampling
    requirements, especially encoder-decoder backbones that rely on divisibility by
    powers of two.

    Attributes:
        multiple (int): Padding base; output height and width are rounded upward to a
            multiple of this value.
    """
    def __init__(self, multiple=32):
        """Initialize the padding module.

        Args:
            multiple (int): Alignment factor used to compute padded dimensions.
        """
        super().__init__()
        self.multiple = multiple

    def forward(self, *inputs):
        """Pad each tensor on the right and bottom to match the target size.

        Args:
            *inputs: One or more tensors, typically an image tensor and a density map.

        Returns:
            list[torch.Tensor]: A list containing each input padded to the computed size.
        """
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
    """Remove right/bottom padding from a tensor to recover the original spatial size.

    This is used after model inference to crop density maps and masks back to the
    original image dimensions before computing counts and segmentation metrics.
    """
    def __init__(self):
        """Initialize the unpadding module."""
        super().__init__()

    def forward(self, padded_tensor, original_shape):
        """Crop a padded tensor back to the original height and width.

        Args:
            padded_tensor (torch.Tensor): A model output or feature map with padded
                spatial dimensions.
            original_shape (tuple | torch.Size): Pair describing the original image shape
                as ``(height, width)``.

        Returns:
            torch.Tensor: Tensor cropped to ``[..., original_h, original_w]``.
        """
        # Extract the original height and width
        org_h, org_w = original_shape[-2:]
        
        # Use ellipsis (...) to cleanly handle any batch or channel dimensions
        return padded_tensor[..., :org_h, :org_w]


class CustomRandomCrop:
    """Pad and randomly crop image-target pairs while maintaining spatial alignment.

    This transform ensures that smaller images are padded to a minimum patch size before
    a random crop is applied to both the RGB image and the density target, preserving
    alignment between the two tensors.
    """
    def __init__(self, patch_size=256):
        """Initialize the crop transform.

        Args:
            patch_size (int): Minimum size used when choosing the random crop window.
        """
        self.patch_size = patch_size

    def __call__(self, img, mask):
        """Pad a sample if needed and return a spatially aligned random crop.

        Args:
            img: Input image tensor or PIL image.
            mask: Corresponding density map or mask tensor.

        Returns:
            tuple: ``(cropped_image, cropped_mask)`` with matching spatial dimensions.
        """
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
    """Apply a constrained set of photometric augmentations without geometric distortion.

    The augmentation pool is restricted to color-space and lighting transforms, which is
    important for density estimation because geometric changes can misalign image and
    target maps.

    Attributes:
        num_ops (int): Number of augmentation operations sampled per call.
        op_pool (torch.nn.ModuleList): Non-geometric transform operations used for
            augmentation.
    """
    def __init__(self, num_ops: int = 2, magnitude: int = 0.3):
        """Initialize the photometric augmentation pool.

        Args:
            num_ops (int): Number of random operations to sample each forward pass.
            magnitude (int): Scale factor controlling augmentation strength.
        """
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
        """Apply a random subset of photometric transforms to an image tensor.

        Args:
            img (torch.Tensor): Input image tensor in normalized ``[0, 1]`` range.

        Returns:
            torch.Tensor: Augmented image tensor with the same spatial shape.
        """
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
    """Provide shared padding helpers for batched evaluation and crop collates.

    The collate functions rely on a common strategy of padding tensors to the next
    multiple of a configured stride so that model outputs remain spatially aligned.
    """
    def __init__(self, multiple):
        """Initialize the collate padding base.

        Args:
            multiple (int): Base multiple used when calculating padded dimensions.
        """
        self.multiple = multiple
        
    def get_padded_dims(self, h, w):
        """Compute padded height and width values rounded up to the next multiple.

        Args:
            h (int): Current height in pixels.
            w (int): Current width in pixels.

        Returns:
            tuple[int, int]: Padded height and width values.
        """
        pad_h = ((h + self.multiple - 1) // self.multiple) * self.multiple
        pad_w = ((w + self.multiple - 1) // self.multiple) * self.multiple
        return pad_h, pad_w
        
    def pad_tensor(self, tensor, target_h, target_w):
        """Pad a tensor on the bottom and right edges to the target shape.

        Args:
            tensor (torch.Tensor): Tensor to pad.
            target_h (int): Desired padded height.
            target_w (int): Desired padded width.

        Returns:
            torch.Tensor: Tensor after right/bottom padding.
        """
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
    """Pad each batch item to the maximum spatial extent in the batch."""
    def __call__(self, batch):
        """Collate a batch while padding images and masks to a common size.

        Args:
            batch: Sequence of ``(image, mask)`` pairs.

        Returns:
            tuple[torch.Tensor, torch.Tensor, list[list[int]]]: Padded images, padded
                masks, and the original spatial sizes for each sample.
        """
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
    """Collate a batch generated from a five-crop evaluation strategy."""
    def __call__(self, batch):
        """Flatten and pad five-crop results into a single collated tensor batch.

        Args:
            batch: Sequence of items where each item contains a tensor stack of five crops.

        Returns:
            tuple[torch.Tensor, torch.Tensor, list[list[int]]]: Flattened and padded crop
                batch plus original crop sizes.
        """
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