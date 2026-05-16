
import os
import sys
sys.path.append('/teamspace/studios/this_studio/workspace/crowd_counting/')
# print(sys.path)
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

# def single_run(experiment_name: str, architecture: str, run_name: str, model_cls: type[CrowdCounter]=CrowdCounter, params: Optional[BaseParams] = None, ckpt_path: Optional[str] = None):

#     print(f"Parameters: {params}")
#     # Create trainer
#     trainer = Experiment(
#         experiment_name=experiment_name,
#         architecture=architecture,
#         model_cls=model_cls,
#     )
#     # Train the model
#     print("Starting training...")
#     model, best_path, val_results, train_results = trainer.fit(params=params, run_name=run_name, ckpt_path=ckpt_path)

#     print(f"\nTraining completed!")
#     print(f"validation: {val_results}")
#     print(f"training: {train_results}")

# def tune(experiment_name: str, architecture: str, run_name: str, n_trials: int = 20):
#     model_cls = CrowdCounter
#     params_cls = BaseParams
#     trainer = Experiment(
#         experiment_name=experiment_name,
#         architecture=architecture,
#         model_cls=model_cls,
#     )
#     trainer.optimize(run_name, params_cls, n_trials)

def main():
    # import os
    # os.environ["TORCH_LOGS"] = "+dynamic"
    experiment_name = "crowd_counting"
    run_name = 'check1'
    # tune(experiment_name, 'hrnet', run_name, n_trials=2)
    params = BaseParams(
        crop_size=256,
        batch_size=2,
        epochs=4,
        lr=0.0006,
        # architecture=architecture,
        # trainable_backbone=False,  # Fine-tune only the head
        # backbone='resnet18',
        # dropout=0.25,
        # aug_factor=0.2,
        # num_ops=4,
        # l2_reg=0.0009,
    )
    architecture = helpers.to_snake_case(params.backbone if params.backbone else params.model_class)
    StandardRunner(CrowdCounter, MLFlowTracker(experiment_name, [architecture, run_name]), params=params).run()
    # single_run(experiment_name, architecture, run_name,params=params)

if __name__ == '__main__':
    main()