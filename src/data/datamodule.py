import timm
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import os
from torchvision import tv_tensors
from src import config
from src.core.params import BaseParams
from src.data.dataset import CustomDataset
from src.data.transform import CustomRandomCrop, FiveCropCollate, PadToMultiple, SafePhotometricRandAugment, DynamicPadCollate
from tqdm import tqdm

class DatasetTransformWrapper(torch.utils.data.Dataset):
    """Apply a transform to dataset samples while optionally caching transformed items.

    This helper wraps an existing dataset subset and applies a per-sample transform on
    access, which is useful for validation and training pipelines that need the same
    image-target preprocessing behavior without changing the base dataset class.

    Attributes:
        subset: Source dataset or subset to wrap.
        transform_fn: Callable used to transform ``(image, target)`` pairs.
        pre_transform (bool): Whether to precompute transformed samples.
        preloaded_data (list): Cache of transformed items when precomputed.
    """
    def __init__(self, subset, transform_fn, pre_transform=False, device='cpu'):
        """Initialize the wrapped dataset.

        Args:
            subset: A dataset-like object providing ``__getitem__`` and ``__len__``.
            transform_fn: Function applied to each image-target sample.
            pre_transform (bool): If ``True``, all transformed samples are cached during
                construction.
            device (str): Legacy device hint retained for compatibility.
        """
        self.subset = subset
        self.transform_fn = transform_fn
        self.pre_transform = pre_transform
        # self.device = device
        self.preloaded_data = []
        
        if self.pre_transform:
            print(f"Pre-transforming {len(subset)} validation samples directly into RAM")
            for i in tqdm(range(len(self.subset))):
                x, y = self.subset[i]
                if self.transform_fn:
                    x, y = self.transform_fn(x, y)
                
                # # Move the fully processed float32 tensors to the GPU
                # x = x.to(self.device)
                # y = y.to(self.device)
                
                self.preloaded_data.append((x, y))
        
    def __getitem__(self, index):
        """Return a transformed sample by index.

        Args:
            index (int): Sample index.

        Returns:
            tuple: The transformed image-target pair.
        """
        if self.pre_transform:
            # 100% CPU-free and PCIe-free fetch during validation
            return self.preloaded_data[index]
            
        x, y = self.subset[index]
        if self.transform_fn:
            x, y = self.transform_fn(x, y)
        return x, y
        
    def __len__(self):
        """Return the number of wrapped samples.

        Returns:
            int: Dataset length.
        """
        return len(self.subset)


class CrowdDataModule(pl.LightningDataModule):
    """Create train, validation, and test dataloaders for crowd-counting experiments.

    The datamodule resolves the configured dataset, builds image-only and image-plus-mask
    transforms, and wraps the dataset with padding and crop logic necessary for density-map
    regression training.

    Attributes:
        data_root (str): Root directory for the dataset.
        params (BaseParams): Parameter bundle controlling pipeline behavior.
        train_joint_augs: Shared geometric transforms applied to both image and mask.
        train_image_augs: Image-only normalization and dtype transforms.
        test_joint_augs: Test-time joint transforms.
        test_image_augs: Test-time image normalization pipeline.
    """
    def __init__(self, data_root=config.DATASET_PATH, params: BaseParams | None = None):
        """Initialize datamodule configuration for the active experiment.

        Args:
            data_root (str): Base directory containing the train/valid/test splits.
            params (BaseParams | None): Experiment parameter object used to resolve model
                backbone statistics and training configuration.
        """
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
            v2.RandomHorizontalFlip(p=0.5),
            # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Applied ONLY to image
            # v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            # v2.RandomAdjustSharpness(sharpness_factor=1.3, p=0.5),
            PadToMultiple(params.padding_multiple)                                       # Applied to BOTH
        ])
        
        self.train_image_augs = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),                  # Applied ONLY to Image
            # SafePhotometricRandAugment(
            #     num_ops=params.num_ops or 4, 
            #     magnitude=int(params.aug_factor)
            # ) if params.aug_factor else v2.Identity(),
            v2.Normalize(mean=mean, std=std)                        # Applied ONLY to Image
        ])

        # ==========================================
        # 2. TEST/VAL PIPELINES (Split into Joint vs Image-Only)
        # ==========================================
        self.test_joint_augs = v2.Compose([
            v2.ToImage(),
        ])
        
        self.test_image_augs = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])

    # Custom wrapper methods to correctly route the data
    def apply_train_transforms(self, img, mask):
        """Apply augmentation and normalization to training samples.

        Args:
            img: Input RGB image tensor or PIL image.
            mask: Density-map target tensor aligned with ``img``.

        Returns:
            tuple: Augmented image and density target pair.
        """
        # 1. Spatial sync: Crop and Pad both equally
        img, mask = self.train_joint_augs(img, mask)
        
        # 2. Color/Stats sync: Mutate the image only
        img = self.train_image_augs(img)
        if mask is not None:
            mask = mask * (self.params.label_scaler or 1)
        
        return img, mask

    def apply_test_transforms(self, img, mask):
        """Apply deterministic inference-time transforms to validation/test samples.

        Args:
            img: Input image tensor.
            mask: Target density map tensor.

        Returns:
            tuple: Normalized image and original density target.
        """
        img, mask = self.test_joint_augs(img, mask)
        img = self.test_image_augs(img)
        # if mask is not None:
        #     mask = mask * (self.params.label_scaler or 1)
        return img, mask

    def apply_five_crop_transforms(self, img, mask):
        """Create a five-crop evaluation batch for a sample and keep masks aligned.

        Args:
            img: Input image tensor.
            mask: Density-target tensor matching the image.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Stacked crop tensors and aligned
                mask crops.
        """
        import torch.nn.functional as F
        
        # 1. Initial base test transforms
        img, mask = self.test_joint_augs(img, mask)
        
        # 2. Extract and format crop_size securely
        crop_size = self.params.crop_size or (512, 512)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
            
        # 3. --- THE FIX: Dynamic Minimum Padding ---
        # If the image is smaller than the requested crop_size, pad it on the bottom/right.
        h, w = img.shape[-2], img.shape[-1]
        pad_bottom = max(0, crop_size[0] - h)
        pad_right = max(0, crop_size[1] - w)
        
        if pad_bottom > 0 or pad_right > 0:
            # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
            img = F.pad(img, (0, pad_right, 0, pad_bottom))
            if mask is not None:
                mask = F.pad(mask, (0, pad_right, 0, pad_bottom))
        
        # 4. Create the deterministic FiveCrop transform
        five_crop = v2.FiveCrop(crop_size)
        
        # 5. Crop the image
        img_crops = five_crop(img)
        
        # 6. Crop the mask separately
        if mask is not None:
            # Strip the tv_tensors.Mask class temporarily
            plain_mask = mask.as_subclass(torch.Tensor)
            
            is_2d = plain_mask.ndim == 2
            if is_2d:
                plain_mask = plain_mask.unsqueeze(0)
                
            mask_crops = five_crop(plain_mask) 
            
            if is_2d:
                mask_crops = tuple(c.squeeze(0) for c in mask_crops)
                
            mask_crops = torch.stack(list(mask_crops))
        else:
            mask_crops = None
            
        # 7. Apply image-only ops on each crop and stack them
        img_crops = torch.stack([self.test_image_augs(c) for c in img_crops])
            
        return img_crops, mask_crops

    def setup(self, stage=None):
        """Configure the training, validation, and test datasets for the current stage.

        Args:
            stage (Optional[str]): Lightning stage name. Supported values are ``"fit"``
                and ``"test"``. If omitted, both are initialized when possible.

        Returns:
            None: The datamodule populates its dataset attributes in place.
        """
        train_path = config.TRAIN_PATH
        test_path = config.TEST_PATH
        # print('train_size:', self.params.train_size)
        # We create separate dataset objects for train and val to use different transforms
        if stage == "fit" or stage is None:
            if getattr(self, 'train_ds', None):
                return 
            val_path = os.path.join(self.data_root, "valid")
    
            # 1. Always load the training set
            full_train_ds = CustomDataset(
                img_dir=os.path.join(train_path, "images"),
                gt_dir=os.path.join(train_path, "ground-truth-npy"),
                preload_to_ram=False,
            )
            
            # 2. Check for the existence of the 'val' folder
            if os.path.exists(val_path):
                print(val_path, "found. Loading validation set from folder.")
                # Load validation dataset from folder
                val_subset = CustomDataset(
                    img_dir=os.path.join(val_path, "images"),
                    gt_dir=os.path.join(val_path, "ground-truth-npy"),
                    preload_to_ram=False
                )
                train_subset = full_train_ds
            else:
                # Perform the random split if 'val' folder does not exist
                train_size = int(0.8 * len(full_train_ds))
                val_size = len(full_train_ds) - train_size
                train_subset, val_subset = random_split(
                    full_train_ds, [train_size, val_size], 
                    generator=torch.Generator().manual_seed(42)
                )
            self.params.train_size = len(train_subset)
            
                # 3. Apply wrappers
            self.train_ds = DatasetTransformWrapper(train_subset, self.apply_train_transforms)
            # for val_dataloader
            if self.params.five_crops:
                self.five_crops_val_ds = DatasetTransformWrapper(val_subset, self.apply_five_crop_transforms, pre_transform=False)
            # for final_val_dataloader
            else:
                self.val_ds = DatasetTransformWrapper(val_subset, self.apply_test_transforms, pre_transform=False)
            self.train_eval_ds = DatasetTransformWrapper(train_subset, self.apply_test_transforms)

        if stage == "test" or stage is None:
            if getattr(self, 'test_ds', None):
                return 
            self.test_ds = CustomDataset(
                img_dir=os.path.join(test_path, "images"),
                gt_dir=os.path.join(test_path, "ground-truth-npy"),
                transform=self.apply_test_transforms # Passes the custom function down
            )

    def train_dataloader(self):
        """Create the training DataLoader for the training split.

        Returns:
            DataLoader: Dataloader that yields batched image-density pairs with random
                shuffling enabled.
        """
        return DataLoader(self.train_ds, batch_size=self.params.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    def val_dataloader(self):
        """Create validation DataLoader using either five-crop or padded batches.

        Returns:
            DataLoader: Validation loader adapted to the configured evaluation mode.
        """
        if self.params.five_crops:
            return DataLoader(
                self.five_crops_val_ds, 
                batch_size=self.params.val_batch_size or 4, 
                num_workers=12, 
                pin_memory=True, 
                collate_fn=FiveCropCollate(self.params.padding_multiple)
            )
        return DataLoader(self.val_ds, batch_size=self.params.val_batch_size or 4, num_workers=12, pin_memory=True, collate_fn=DynamicPadCollate(self.params.padding_multiple))

    def test_dataloader(self):
        """Create the inference-time DataLoader for the test split.

        Returns:
            DataLoader: Test loader that batches images and density maps with padding.
        """
        return DataLoader(self.test_ds or self.val_ds, batch_size=self.params.val_batch_size or 4, num_workers=4, pin_memory=True, collate_fn=DynamicPadCollate(self.params.padding_multiple))
    
    def train_eval_dataloader(self):
        """Create a clean DataLoader for evaluating the training split after fitting.

        Returns:
            DataLoader: Non-shuffled evaluation loader for training-set metrics.
        """
        return DataLoader(self.train_eval_ds, batch_size=self.params.val_batch_size or 4, shuffle=False, num_workers=12, collate_fn=DynamicPadCollate(self.params.padding_multiple))
    