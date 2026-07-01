import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from PIL import Image
import h5py
import glob
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_dir, h5_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.h5_paths = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
        print(len(self.img_paths), "images found in", img_dir)
        print(len(self.h5_paths), "h5 files found in", h5_dir)
        assert len(self.img_paths) == len(self.h5_paths), "Mismatch between images and h5 files"
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Raw Data
        img = Image.open(self.img_paths[idx]).convert('RGB')
        with h5py.File(self.h5_paths[idx], 'r') as hf:
            target = np.asarray(hf['density']).astype(np.float32)
        
        # 2. Wrap in TVTensors 
        # Wrapping as 'Mask' ensures spatial transforms (Crop/Resize) apply, 
        # but pixel-value transforms (Normalize) are skipped.
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(torch.from_numpy(target).unsqueeze(0))

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
        # NOTE: This assumes CustomDataset stores a list of paths in `self.img_paths`.
        # If your variable is named differently (e.g., self.images), update it here.
        img_path = dataset.img_paths[real_index]
        
        # 2. Get the ground truth mask by calling the underlying dataset
        # We discard the loaded image `_` because we only want the path and mask
        _, mask = dataset[real_index]
        
        return img_path, mask
        
    def __len__(self):
        return len(self.subset)