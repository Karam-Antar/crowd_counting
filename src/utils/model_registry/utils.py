"""Shared utilities and data structures for model registry."""

import shutil
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from lightning.pytorch.loggers import LitLogger
import torch
import urllib.parse

from src import config
from src.core.params import BaseParams
from src.utils import helpers
from src.utils.experiment_trackers import BaseTracker
from src.utils.model_persistence import save_model


# --- Data Structures ---

@dataclass
class ModelPayload:
    """Encapsulates all artifacts and metadata produced by a training run."""
    model: torch.nn.Module
    tracker: BaseTracker
    params: BaseParams
    metrics: dict
    ckpt_path: Optional[str] = None
    code_artifacts: Optional[dict] = None


# --- Shared Preparation Logic (Backend Agnostic) ---

def add_metadata(artifacts_path: Path, params: BaseParams, experiment_name: str, metrics: dict, empty_files=False):
    config_file_path = artifacts_path / "model_config.json"
    best_metrics = {k: v for k, v in metrics.items() if str(k).casefold().startswith('best')}
    params.to_json(config_file_path, meta={'experiment': experiment_name, 'metrics': best_metrics})
    if empty_files:
        helpers.create_empty_text_files(
            artifacts_path, 
            [f'experiment={urllib.parse.quote(experiment_name, safe="=")}', *helpers.generate_file_names(best_metrics)]
        )

def prepare_temp_dir(artifacts_path: Path, payload: ModelPayload, experiment_name: str):
    """Prepares a temporary directory with the model, checkpoints, and metadata."""
    os.makedirs(artifacts_path, exist_ok=True)
    
    # 1. Add Config and Metadata
    add_metadata(artifacts_path, payload.params, experiment_name, payload.metrics)
    
    # 2. Save PyTorch Model
    # if isinstance(payload.tracker.logger, LitLogger):
    
    # 3. Copy Checkpoint if it exists
    if payload.ckpt_path: 
        shutil.copy(payload.ckpt_path, artifacts_path / payload.ckpt_path.split('/')[-1])
        
    # 4. Handle Code Artifacts (Zip Git or copy specific files)
    if payload.code_artifacts is None:
        code_path = Path(f'{artifacts_path}/code.tar.gz')
        code_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(f"git ls-files --cached --others --exclude-standard | tar -czvf '{code_path.as_posix()}' -T -", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create zip: {e}")
        return
        
    for artifact_name, source_path in payload.code_artifacts.items():
        source_path = Path(source_path)
        destination = artifacts_path / artifact_name
        if source_path.is_dir():
            shutil.copytree(source_path, destination, dirs_exist_ok=True) 
        else:
            shutil.copy(source_path, destination)
