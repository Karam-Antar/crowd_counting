from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import optuna

from src.utils import helpers

@dataclass
class BaseParams:
    """Base parameter class."""
    image_size: Optional[int] = None
    crop_size: Optional[int] = None
    backbone: Optional[str] = None
    model_class: str = 'CrowdCounter'
    batch_size: int = 16
    epochs: int = 15
    aug_factor: Optional[float] = None
    padding_multiple: int = 32
    num_ops: Optional[int] = None
    train_size: Optional[int] = None
    l2_reg: Optional[float] = None
    trainable_backbone: bool = False
    neck_out_channels: int = 64
    backbone: str = 'hrnet_w18'
    unfrozen_blocks: Optional[tuple] = None
    dropout: Optional[float] = 0.25
    
    lr: float = 1e-3
    lr_schedule: Optional[str] = None

    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    optimizer_config: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)
    suggested_params: bool = False
    min_lr_pct: Optional[float] = None

    @classmethod
    def suggest(cls, trial: optuna.Trial) -> dict:
        return dict(
            image_size=trial.suggest_categorical("image_size", [128, 160, 180]),
            # crop_size=trial.suggest_categorical("crop_size", [128, 160, 180]),
            batch_size=trial.suggest_categorical("batch_size", [1, 1, 1]),
            lr=trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            l2_reg=trial.suggest_float("l2_reg", 1e-4, 1e-3, log=True),
            # min_lr_pct=trial.suggest_float("min_lr_pct", 0.01, 0.2),
            aug_factor=trial.suggest_float("aug_factor", 0.05, 0.15),
            suggested_params=True
        )
    
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