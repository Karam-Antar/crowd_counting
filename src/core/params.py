from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import optuna

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

    # Architecture tweaks
    backbone: str = 'efficientnet-b0'
    model_class: str = 'Unet'
    backbone_weights: str = 'imagenet'
    decoder_attention_type: Optional[str] = None
    trainable_backbone: bool = False
    neck_out_channels: Optional[int] = None
    dropout: Optional[float] = None

    # Training hardware/flow limits
    batch_size: int = 16

    # Optimization & Regularization
    lr: float = 0.00065
    l2_reg: Optional[float] = 1e-4
    lr_schedule: Optional[str] = None
    # min_lr_pct: Optional[float] = None
    monitor_metric: str = 'val_nae'
    loss_function: str = 'mse'
    grad_accumulation: int = 1
    grad_clip: Optional[float] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    suggested_params: bool = False

    # Properties not present in the suggest method (placed last)
    epochs: int = 60
    padding_multiple: int = 32
    dataset: str = config.DATASET_PATH.split('/')[-1]
    train_size: Optional[int] = None
    unfrozen_blocks: Optional[tuple] = None
    optimizer_config: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)

    @classmethod
    def suggest(cls, trial: optuna.Trial) -> "BaseParams":
        suggested = dict(
            # Image & Data Augmentation
            # image_size=trial.suggest_categorical("image_size", [512, 768, 1024]), # Adjust based on dataset
            crop_size=trial.suggest_categorical("crop_size", [128, 256, 512]),
            # aug_factor=trial.suggest_float("aug_factor", 0.0, 0.3),
            # num_ops=trial.suggest_int("num_ops", 1, 4),

            # Architecture tweaks
            backbone=trial.suggest_categorical("backbone", ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'resnet34', 'resnet50']),
            # trainable_backbone=trial.suggest_categorical("trainable_backbone", [True, False]),
            # neck_out_channels=trial.suggest_categorical("neck_out_channels", [32, 64, 128]),
            # dropout=trial.suggest_float("dropout", 0.0, 0.5),

            # Training hardware/flow limits
            batch_size=trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64]),     # Kept small for high-res density maps

            # Optimization & Regularization
            # lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            # l2_reg=trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True),
            # lr_schedule=trial.suggest_categorical("lr_schedule", ['clipped_exp']),
            scheduler_kwargs={
                'decay_rate': trial.suggest_float("decay_rate", 0.88, 0.97),
                'min_lr_pct': trial.suggest_float("min_lr_pct", 0.01, 0.1),
            },

            # Flag indicating this was generated via Optuna
            suggested_params=True
        )
        unfrozen = trial.suggest_int("unfrozen", 0, 4)
        suggested['unfrozen_blocks'] = registries.UNFROZEN.get(next(e for e in registries.UNFROZEN.keys() if str(e).startswith(suggested['backbone'])), [])[:unfrozen]
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