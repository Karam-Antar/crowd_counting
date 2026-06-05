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