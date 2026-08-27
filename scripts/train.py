
import sys
sys.path.append('/home/jl_fs/workspace/projects/crowd_counting')
# print(sys.path)
from src.experiment.optuna_tuner import OptunaTuner
from src.utils.experiment_trackers import MLFlowTracker
from typing import Optional
import torch
import lightning.pytorch as pl
# import mlflow
from pathlib import Path
from src.models.lit_model import BaseLitModel
from src.core.params import BaseParams
from src.experiment import StandardRunner
from src.models.model import CrowdCounter
from src.data.datamodule import CrowdDataModule
import mlflow
# Set random seeds for reproducibility
from src import config
from src.utils import helpers
config.set_seed()

print(f"PyTorch: {torch.__version__}")
print(f"Lightning: {pl.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# def get_datamodule(params: BaseParams):
#     datamodule = CrowdDataModule(params=params)
#     # datamodule.setup(stage='base')
#     return datamodule


# def main():
# import os
# os.environ["TORCH_LOGS"] = "+dynamic"
mlflow.config.enable_async_logging(True)
experiment_name = "crowd_counting"
run_name = None
# study_name = 'check1'
params = BaseParams(
    model_class='MAnet',
    backbone='tu-convnext_base',
    # trainable_backbone=True,
    # decoder_attention_type='scse',
    backbone_weights='imagenet',
    # unfrozen_blocks=('blocks.15',),
    crop_size=480,
    batch_size=8,
    val_batch_size=1,
    stop_patience=35,
    # check_val_every_n_epoch=10,
    dropout=0.2,
    decoder_out_channels=128,
    loss_function='mask_mse_ssim',
    ssim_weight=0.65,
    mask_loss_weight=0.85,
    mask_loss_alpha=0.77,
    mask_loss_gamma=3.8,
    gt_mask_threshold=0,
    huber_delta=5,
    # k_threshold=70,
    # aug_factor=0.18,
    # num_ops=4,
    epochs=60,
    # l2_reg=0.0009,
    lr=0.00065,
    lr_schedule='clipped_exp',
    # grad_accumulation=16,
    scheduler_kwargs={
        'decay_rate': 0.945,
        'min_lr_pct': 0.01,
    },
)
# architecture = helpers.to_snake_case(params.backbone if params.backbone else params.model_class)
payload, val_results, train_results = StandardRunner(CrowdCounter, MLFlowTracker(experiment_name, run_name), params=params, monitor_metric='val_mae_mbe').run()
print(f"\nTraining completed!")
print(f"validation: {val_results}")
print(f"training: {train_results}")
# OptunaTuner(experiment_name,  CrowdCounter, study_name, n_trials=2).run()

# if __name__ == '__main__':
#     main()