import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from PIL import Image
import glob
import os
import numpy as np
# REMOVED: import h5py

class CustomDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        print(len(self.img_paths), "images found in", img_dir)
        print(len(self.gt_paths), "gt files found in", gt_dir)
        assert len(self.img_paths) == len(self.gt_paths), "Mismatch between images and npy files"
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Raw Data
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        # FAST NPY LOADING: Replaced h5py with np.load memory mapping
        target_array = np.load(self.gt_paths[idx], mmap_mode='r')
        
        # 2. Wrap in TVTensors 
        img = tv_tensors.Image(img)
        # .copy() is strictly required here because mmap arrays are read-only, 
        # and PyTorch throws errors if you create a tensor from read-only memory.
        target = tv_tensors.Mask(torch.from_numpy(target_array.copy()).float().unsqueeze(0))

        # 3. Apply Unified Transform
        if self.transform:
            img, target = self.transform(img, target)

        return img, target


class EnsemblePathWrapper(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset
        
    def __getitem__(self, index):
        # Handle torch.utils.data.Subset (created by random_split)
        if isinstance(self.subset, torch.utils.data.Subset):
            dataset = self.subset.dataset
            real_index = self.subset.indices[index]
        else:
            dataset = self.subset
            real_index = index
            
        # 1. Get the path directly from your CustomDataset
        img_path = dataset.img_paths[real_index]
        
        # 2. Get the ground truth mask by calling the underlying dataset
        _, mask = dataset[real_index]
        
        return img_path, mask
        
    def __len__(self):
        return len(self.subset)