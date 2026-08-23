import sys

sys.path.append('/teamspace/studios/this_studio/workspace/crowd_counting/')

import lightning.pytorch as pl
import torch

from src import config
from src.experiment.optuna_tuner import OptunaTuner
from src.models.model import CrowdCounter
from src.utils.experiment_trackers import MLFlowTracker
from src.utils.model_registry import MLFlowRegistry


EXPERIMENT_NAME = "crowd_counting"
STUDY_NAME_SUFFIX = "baseline"
N_TRIALS = 2


def main():
    config.set_seed()

    print(f"PyTorch: {torch.__version__}")
    print(f"Lightning: {pl.__version__}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tuner = OptunaTuner(
        experiment=EXPERIMENT_NAME,
        model_cls=CrowdCounter,
        study_name_suffix=STUDY_NAME_SUFFIX,
        tracker_cls=MLFlowTracker,
        registry_cls=MLFlowRegistry,
        n_trials=N_TRIALS,
    )
    best_payload, study = tuner.run()

    print("\nOptimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value ({study.direction.name}): {study.best_value}")
    if best_payload is not None:
        print("Best model uploaded to the registry.")


if __name__ == "__main__":
    main()
