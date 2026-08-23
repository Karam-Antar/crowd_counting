from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from src import config
from src.utils import helpers, registries

@dataclass
class BaseParams:
    """Base parameter class."""
    # Image & Data Augmentation
    image_size: Optional[int] = None
    crop_size: Optional[int] = None
    label_scaler: Optional[int] = config.LABEL_SCALER
    aug_factor: Optional[float] = None
    num_ops: Optional[int] = None
    five_crops: bool = False

    # Architecture tweaks
    backbone: str = 'efficientnet-b0'
    model_class: str = 'Unet'
    backbone_weights: str = 'imagenet'
    decoder_attention_type: Optional[str] = None
    trainable_backbone: bool = False
    neck_out_channels: Optional[int] = None
    decoder_out_channels: Optional[int] = None
    dropout: Optional[float] = None

    # Training hardware/flow limits
    batch_size: int = 16
    val_batch_size: int = 32

    # Optimization & Regularization
    lr: float = 0.00065
    l2_reg: Optional[float] = 1e-4
    lr_schedule: Optional[str] = None
    ssim_weight: Optional[float] = None
    mask_loss_weight: Optional[float] = None
    huber_delta: Optional[float] = None
    mask_loss_alpha: Optional[float] = None
    mask_loss_gamma: Optional[float] = None
    k_threshold: Optional[float] = None
    seg_threshold: Optional[float] = None
    gt_mask_threshold: Optional[float] = None
    # min_lr_pct: Optional[float] = None
    monitor_metric: str = 'val_nae'
    loss_function: str = 'mse'
    grad_accumulation: int = 1
    grad_clip: Optional[float] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    suggested_params: bool = False

    # Properties not present in the suggest method (placed last)
    epochs: int = 1
    stop_patience: int = 45
    check_val_every_n_epoch: Optional[int] = 1
    padding_multiple: int = 32
    dataset: str = config.DATASET_PATH.split('/')[-1]
    train_size: Optional[int] = None
    unfrozen_blocks: Optional[tuple] = None
    optimizer_config: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)

    @classmethod
    def suggest(cls, trial) -> "BaseParams":
        suggested = dict(
            # 1. Architecture & Base Setup (Locked to your current experiment)
            model_class=trial.suggest_categorical('model_class', ['MAnet', 'Unet', 'UnetPlusPlus']),
            backbone=trial.suggest_categorical('backbone', ['tu-convnext_small', 'tu-convnext_base', 'tu-convnextv2_small', 'tu-convnextv2_base', 'efficientnet-b2', 'efficientnet-b4', 'efficientnet-b6']), 
            backbone_weights='imagenet',
            decoder_out_channels=trial.suggest_categorical("decoder_out_channels", [64, 128, 256]),
            
            # 2. Memory & Scaling
            # ConvNeXt is heavy. Keep batch sizes bounded so you don't OOM (Out Of Memory)
            crop_size=trial.suggest_categorical("crop_size", [384, 480, 512]),
            batch_size=trial.suggest_categorical("batch_size", [4, 8, 12]), 
            val_batch_size=1,

            # 3. Learning Rate & Scheduling
            # Log scale is best for learning rates. Searches around your 0.00065 base.
            lr=trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            lr_schedule='clipped_exp',
            scheduler_kwargs={
                'decay_rate': trial.suggest_float("decay_rate", 0.90, 0.99),
                'min_lr_pct': trial.suggest_float("min_lr_pct", 0.005, 0.05),
            },

            # 4. Composite Loss Parameters 
            # Centered around your baseline: ssim=0.3, alpha=0.81, gamma=3.17, delta=5
            loss_function='mask_mse_ssim',
            ssim_weight=trial.suggest_float("ssim_weight", 0.1, 0.5),
            mask_loss_weight=trial.suggest_float("mask_loss_weight", 0.5, 1.0),  # Anchor weight (keep fixed, tune others relative to this)
            mask_loss_alpha=trial.suggest_float("mask_loss_alpha", 0.6, 0.95),
            mask_loss_gamma=trial.suggest_float("mask_loss_gamma", 2.0, 4.0),
            huber_delta=trial.suggest_float("huber_delta", 2.0, 13),
            gt_mask_threshold=0,

            # 5. Regularization (Optional but recommended for ConvNeXt)
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            l2_reg=trial.suggest_float("l2_reg", 1e-4, 1e-3, log=True),

            suggested_params=True
        )

        # 6. Unfrozen Blocks Logic
        # Wrapped in a try/except so Optuna doesn't crash if 'tu-convnext_small' 
        # isn't explicitly mapped in your registries.UNFROZEN dictionary yet.
        unfrozen = trial.suggest_int("unfrozen", 0, 4)
        try:
            backbone_key = next(e for e in registries.UNFROZEN.keys() if str(e).startswith(suggested['backbone']))
            suggested['unfrozen_blocks'] = registries.UNFROZEN.get(backbone_key, [])[:unfrozen]
        except StopIteration:
            suggested['unfrozen_blocks'] = None

        return cls(**suggested)
    
    def to_dict(self, flatten=False, to_str=False, nested=True) -> Dict[str, Any]:
        params_dict = {'params': asdict(self)} if nested else asdict(self)
        if flatten:
            params_dict = helpers.flatten_dict(params_dict, to_str=to_str)
        if not nested and to_str:
            params_dict = helpers.dict_to_str(params_dict)
        return params_dict
    
    def to_json(self, file_path: str, flatten=False, to_str=False, indent=4, meta={}):
        """Saves the object data to a JSON file."""
        import json
        data = self.to_dict(flatten=flatten, to_str=to_str)
        data.update({'meta': meta})
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
    
    @classmethod
    def from_dict(cls, params_dict: dict):
        if params_dict.get('params', None):
            return cls(**params_dict['params'])
        params_dict = helpers.unflatten_dict({k: v for k, v in params_dict.items() if str(k).casefold().startswith('params')}).get('params', params_dict)
        return cls(**params_dict)
    
    @classmethod
    def from_json(cls, file_path: str):
        """Reads a JSON file and returns an instance of the class."""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)