from pathlib import Path
import sys
from dotenv import find_dotenv, load_dotenv

# 1. Load environment variables dynamically from .env
load_dotenv(find_dotenv())

# 2. Add project root dynamically to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 3. Standard imports
from typing import Optional
import torch
import lightning.pytorch as pl
import mlflow

from src import config
from src.core.params import BaseParams
from src.experiment import StandardRunner
from src.experiment.optuna_tuner import OptunaTuner
from src.models.lit_model import BaseLitModel
from src.models.model import CrowdCounter
from src.utils import helpers
from src.utils.experiment_trackers import MLFlowTracker

# Set random seeds for reproducibility
config.set_seed()

print(f"PyTorch: {torch.__version__}")
print(f"Lightning: {pl.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def main():
    mlflow.config.enable_async_logging(True)
    experiment_name = "crowd_counting"
    run_name = None

    params = BaseParams(
        model_class='MAnet',
        backbone='tu-convnext_base',
        backbone_weights='imagenet',
        crop_size=480,
        batch_size=8,
        val_batch_size=1,
        stop_patience=35,
        dropout=0.2,
        decoder_out_channels=128,
        loss_function='mask_mse_ssim',
        use_count_loss=True,
        ssim_weight=1 * 2.8,
        mse_weight=0.000001 * 1.5,
        mask_loss_weight=10 * 0.4,
        mask_loss_alpha=0.77,
        mask_loss_gamma=3.8,
        gt_mask_threshold=0,
        huber_delta=2,
        epochs=60,
        lr=0.00065,
        lr_schedule='clipped_exp',
        scheduler_kwargs={
            'decay_rate': 0.945,
            'min_lr_pct': 0.01,
        },
    )

    payload, val_results, train_results = StandardRunner(
        CrowdCounter, 
        MLFlowTracker(experiment_name, run_name), 
        params=params, 
        monitor_metric='val_mae_mbe'
    ).run()

    print("\nTraining completed!")
    print(f"validation: {val_results}")
    print(f"training: {train_results}")


if __name__ == '__main__':
    main()