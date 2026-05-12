import timm
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import os

from src import config
from src.data.dataset import ShanghaiTechDataset

# Helper class to apply transforms to random_split subsets
class DatasetTransformWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y
        
    def __len__(self):
        return len(self.subset)



class CrowdDataModule(pl.LightningDataModule):
    def __init__(self, data_root=config.DATASET_PATH, params=None):
        super().__init__()
        self.data_root = data_root
        self.batch_size = params.batch_size
        self.patch_size = params.crop_size

        # Unified Train Pipeline: Everything happens here
        data_config = timm.data.resolve_data_config({}, model=params.backbone)

        # Extract stats
        mean = data_config['mean']
        std = data_config['std']
        self.train_transform = v2.Compose([
            v2.ToImage(),                            # Convert PIL to Tensor
            v2.RandomCrop(size=(self.patch_size, self.patch_size)),
            v2.RandAugment(
                    num_ops=params.num_ops or 4, 
                    magnitude=int(params.aug_factor * 30)
                ),
            v2.ToDtype(torch.float32, scale=True),   # Scale image to [0, 1], skip Mask
            v2.Normalize(mean=mean, std=std)
        ])

        # Unified Test/Val Pipeline: No cropping, just normalization
        self.test_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])

    def setup(self, stage=None):
        train_path = os.path.join(self.data_root, "train_data")
        test_path = os.path.join(self.data_root, "test_data")

        # We create separate dataset objects for train and val to use different transforms
        if stage == "fit" or stage is None:
            full_train_ds = ShanghaiTechDataset(
                img_dir=os.path.join(train_path, "images"),
                h5_dir=os.path.join(train_path, "ground-truth-h5")
            )
            
            # Split indices
            train_size = int(0.8 * len(full_train_ds))
            val_size = len(full_train_ds) - train_size
            train_subset, val_subset = random_split(
                full_train_ds, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
            self.params.train_size = train_size
            # Re-wrap subsets to apply specific transforms
            self.train_ds = DatasetTransformWrapper(train_subset, self.train_transform)
            self.val_ds = DatasetTransformWrapper(val_subset, self.test_transform)

        if stage == "test" or stage is None:
            self.test_ds = ShanghaiTechDataset(
                img_dir=os.path.join(test_path, "images"),
                h5_dir=os.path.join(test_path, "ground-truth-h5"),
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        # BS=1 is standard for crowd counting validation on full images
        return DataLoader(self.val_ds, batch_size=1, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=4)