import timm
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import os

from src import config
from src.core.params import BaseParams
from src.data.dataset import ShanghaiTechDataset
from src.data.transform import CustomRandomCrop, PadToMultiple, SafePhotometricRandAugment

# Helper class to apply transforms to random_split subsets
class DatasetTransformWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform_fn):
        self.subset = subset
        self.transform_fn = transform_fn
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        # Pass both x and y to our custom transform function
        if self.transform_fn:
            x, y = self.transform_fn(x, y)
        return x, y
        
    def __len__(self):
        return len(self.subset)


class CrowdDataModule(pl.LightningDataModule):
    def __init__(self, data_root=config.DATASET_PATH, params: BaseParams | None = None):
        super().__init__()
        self.data_root = data_root
        self.params = params
        
        data_config = timm.data.resolve_data_config({}, model=params.backbone)
        mean = data_config['mean']
        std = data_config['std']

        # ==========================================
        # 1. TRAIN PIPELINES (Split into Joint vs Image-Only)
        # ==========================================
        self.train_joint_augs = v2.Compose([
            v2.ToImage(),                                           # Convert PIL to Tensor
            CustomRandomCrop(params.crop_size) if params.crop_size else v2.Identity(), # Applied to BOTH
            PadToMultiple(params.padding_multiple)                                       # Applied to BOTH
        ])
        
        self.train_image_augs = v2.Compose([
            SafePhotometricRandAugment(
                num_ops=params.num_ops or 4, 
                magnitude=int(params.aug_factor * 30)
            ) if params.aug_factor else v2.Identity(),                                                      # Applied ONLY to Image
            v2.ToDtype(torch.float32, scale=True),                  # Applied ONLY to Image
            v2.Normalize(mean=mean, std=std)                        # Applied ONLY to Image
        ])

        # ==========================================
        # 2. TEST/VAL PIPELINES (Split into Joint vs Image-Only)
        # ==========================================
        self.test_joint_augs = v2.Compose([
            v2.ToImage(),
            PadToMultiple(params.padding_multiple)
        ])
        
        self.test_image_augs = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])

    # Custom wrapper methods to correctly route the data
    def apply_train_transforms(self, img, mask):
        # 1. Spatial sync: Crop and Pad both equally
        img, mask = self.train_joint_augs(img, mask)
        
        # 2. Color/Stats sync: Mutate the image only
        img = self.train_image_augs(img)
        if mask is not None:
            mask = mask * (self.params.label_scaler or 1)
        
        return img, mask

    def apply_test_transforms(self, img, mask):
        img, mask = self.test_joint_augs(img, mask)
        img = self.test_image_augs(img)
        if mask is not None:
            mask = mask * (self.params.label_scaler or 1)
        return img, mask

    def setup(self, stage=None):
        train_path = os.path.join(self.data_root, "train_data")
        test_path = os.path.join(self.data_root, "test_data")
        # print('train_size:', self.params.train_size)
        # We create separate dataset objects for train and val to use different transforms
        if stage == "fit" or stage is None:
            if getattr(self, 'train_ds', None):
                return 
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
            
            # Use our custom apply_*_transforms methods here!
            self.train_ds = DatasetTransformWrapper(train_subset, self.apply_train_transforms)
            self.val_ds = DatasetTransformWrapper(val_subset, self.apply_test_transforms)
            self.train_eval_ds = DatasetTransformWrapper(train_subset, self.apply_test_transforms)

        if stage == "test" or stage is None:
            if getattr(self, 'test_ds', None):
                return 
            self.test_ds = ShanghaiTechDataset(
                img_dir=os.path.join(test_path, "images"),
                h5_dir=os.path.join(test_path, "ground-truth-h5"),
                transform=self.apply_test_transforms # Passes the custom function down
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.params.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        # BS=1 is standard for crowd counting validation on full images
        return DataLoader(self.val_ds, batch_size=1, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=4)
    
    def train_eval_dataloader(self):
        """Used ONLY for evaluating the training set cleanly."""
        return DataLoader(self.train_eval_ds, batch_size=1, shuffle=False, num_workers=4)