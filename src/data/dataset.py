import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from PIL import Image
import glob
import os
import numpy as np
from tqdm import tqdm
# REMOVED: import h5py

class CustomDataset(Dataset):
    """Dataset for loading image-density pairs from disk.

    The dataset expects a folder of RGB images and a parallel folder of NumPy density
    maps with matching basenames. It is used for crowd-counting training and validation
    data ingestion and optionally supports loading a subset into memory.

    Attributes:
        img_paths (list[str]): Sorted paths to image files.
        gt_paths (list[str]): Sorted paths to NumPy ground-truth density maps.
        transform: Optional callable used to transform image-target pairs after loading.
        preload_to_ram (bool): Whether to cache samples in memory during initialization.
        data (list[tuple]): Optional in-memory cache of preloaded samples.
    """
    def __init__(self, img_dir, gt_dir, transform=None, preload_to_ram=False):
        """Initialize the dataset by scanning paired image and target directories.

        Args:
            img_dir (str): Directory containing RGB input images.
            gt_dir (str): Directory containing NumPy ground-truth density maps.
            transform: Optional transform applied to each image-target pair.
            preload_to_ram (bool): If ``True``, load all samples into memory during
                initialization instead of reading from disk per item.

        Raises:
            AssertionError: If the image and target counts do not match exactly.
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        print(len(self.img_paths), "images found in", img_dir)
        print(len(self.gt_paths), "gt files found in", gt_dir)
        assert len(self.img_paths) == len(self.gt_paths), "Mismatch between images and npy files"
        
        self.transform = transform
        self.preload_to_ram = preload_to_ram
        self.data = []

        if self.preload_to_ram:
            print(f"Preloading {len(self.img_paths)} samples to RAM. Watch your memory usage!")
            for img_p, gt_p in tqdm(zip(self.img_paths, self.gt_paths), total=len(self.img_paths)):
                # 1. Load Image, convert to tensor immediately
                with Image.open(img_p) as img:
                    img_tensor = tv_tensors.Image(img.convert('RGB'))
                
                # 2. Load NPY fully into RAM (removed mmap_mode='r')
                target_array = np.load(gt_p) 
                target_tensor = tv_tensors.Mask(torch.from_numpy(target_array).float().unsqueeze(0))
                
                self.data.append((img_tensor, target_tensor))

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Load and optionally transform a single image-ground-truth pair.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: A pair of ``(image, target)`` tensors, where the target is the
                density map mask for crowd counting.

        Raises:
            IndexError: If the index is outside the dataset range.
        """
        if self.preload_to_ram:
            # Retrieve directly from memory
            img, target = self.data[idx]
        else:
            # Fallback to standard disk read
            img = Image.open(self.img_paths[idx]).convert('RGB')
            img = tv_tensors.Image(img)
            
            target_array = np.load(self.gt_paths[idx], mmap_mode='r')
            target = tv_tensors.Mask(torch.from_numpy(target_array.copy()).float().unsqueeze(0))

        # Apply Unified Transform
        if self.transform:
            img, target = self.transform(img, target)

        return img, target