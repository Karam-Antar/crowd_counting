import os
import random
import numpy as np
import torch

SEED = 42
LABEL_SCALER = 1000
# CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
MODEL_SAVE_PATH = "/teamspace/studios/this_studio/workspace/crowd_counting/trained_models"
LOG_DIR = "./logs"
MLFLOW_DB_URL = os.getenv('MLFLOW_DB_URL', None)
TORCH_HOME = os.getenv('TORCH_HOME')
OPTUNA_DB_URL = os.getenv('OPTUNA_DB_URL')
DATA_HOME = os.getenv('DATA_HOME', '/teamspace/lightning_storage/datasets')
DATASET_PATH = f'{DATA_HOME}/ShanghaiTech/part_A'
PIP_REQUIREMENTS = [
    "torch",
    "torchvision",
    "lightning",
    "matplotlib",
    "numpy",
    "pandas",
    "Pillow",
    'timm',
    'torchmetrics',
    # 'optuna',
]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed():
    """Set reproducible seed."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)