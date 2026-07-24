import os
import random
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# 1. Locate the .env file dynamically (searches upward from current working directory)
# This fixes the script vs. notebook execution location issue.
env_path = find_dotenv()
if not env_path:
    raise FileNotFoundError("Could not find .env file. Please create one at the project root.")

load_dotenv(env_path)

# 2. Establish an absolute anchor for the project root based on where .env was found
PROJECT_ROOT = Path(env_path).parent

SEED = 42
LABEL_SCALER = 1000

# 3. Build paths relative to your dynamic PROJECT_ROOT
MODEL_SAVE_PATH = PROJECT_ROOT / "trained_models"
LOG_DIR = PROJECT_ROOT / "logs"

MLFLOW_DB_URL = os.getenv('MLFLOW_DB_URL')
TORCH_HOME = os.getenv('TORCH_HOME')
OPTUNA_DB_URL = os.getenv('OPTUNA_DB_URL')

# If DATA_HOME is in the .env, use it. Otherwise, default to PROJECT_ROOT/datasets
DATA_HOME = Path(os.getenv('DATA_HOME', PROJECT_ROOT / 'datasets'))

SHANGHAI_PATH = DATA_HOME / 'ShanghaiTech' / 'part_A'
JHU_PATH = DATA_HOME / 'jhu-crowd-pp-v2'
DATASET_PATH = JHU_PATH

TRAIN_PATH = DATASET_PATH / ('train' if DATASET_PATH == JHU_PATH else 'train_data')
TEST_PATH = DATASET_PATH / ('test' if DATASET_PATH == JHU_PATH else 'test_data')

# Replaced your hardcoded '/home/jl_fs/workspace/...' with PROJECT_ROOT
SERVE_REQUIREMENTS_PATH = PROJECT_ROOT / 'requirements-serve.txt'
SERVE_CODE_PATH = PROJECT_ROOT / 'src'
PYFUNC_MODEL_PATH = SERVE_CODE_PATH / 'core' / 'pyfunc.py'
PARAMS_SAVE_FILENAME = 'model_config.json'


DATA_HOME = str(DATA_HOME)
DATASET_PATH = str(DATASET_PATH)
SERVE_REQUIREMENTS_PATH = str(SERVE_REQUIREMENTS_PATH)
SERVE_CODE_PATH = str(SERVE_CODE_PATH)
PYFUNC_MODEL_PATH = str(PYFUNC_MODEL_PATH)
PARAMS_SAVE_FILENAME = str(PARAMS_SAVE_FILENAME)
MODEL_SAVE_PATH = str(MODEL_SAVE_PATH)
LOG_DIR = str(LOG_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed():
    """Set reproducible seed."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)