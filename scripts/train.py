
import sys
sys.path.append('/teamspace/studios/this_studio/workspace/crowd_counting/')
# print(sys.path)
from src.experiment.optuna_tuner import OptunaTuner
from src.utils.experiment_trackers import MLFlowTracker
import torch
import lightning.pytorch as pl
# import mlflow
from src.core.params import BaseParams
from src.experiment import StandardRunner
from src.models.model import CrowdCounter

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


def main():
    # import os
    # os.environ["TORCH_LOGS"] = "+dynamic"
    experiment_name = "crowd_counting"
    run_name = None
    # study_name = 'check1'
    params = BaseParams(
        model_class='Unet',
        backbone='efficientnet-b2',
        # unfrozen_blocks=('stage4',),
        crop_size=640,
        batch_size=16,
        # aug_factor=0.18,
        # num_ops=4,
        epochs=35,
        # l2_reg=0.0009,
        lr=0.00065,
        lr_schedule='clipped_exp',
        # grad_accumulation=16,
        scheduler_kwargs={
            'decay_rate': 0.96,
            'min_lr_pct': 0.01,
        },
    )
    # architecture = helpers.to_snake_case(params.backbone if params.backbone else params.model_class)
    payload, val_results, train_results = StandardRunner(CrowdCounter, MLFlowTracker(experiment_name, run_name), params=params).run()
    print(f"\nTraining completed!")
    print(f"validation: {val_results}")
    print(f"training: {train_results}")
    # OptunaTuner(experiment_name,  CrowdCounter, study_name, n_trials=2).run()

if __name__ == '__main__':
    main()