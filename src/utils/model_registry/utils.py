"""Shared utilities and data structures for model registry."""

import shutil
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import zipfile
from lightning.pytorch.loggers import LitLogger
import torch
import urllib.parse

from torch import monitor

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
    monitor_metric: str
    monitor_mode: str
    ckpt_path: Optional[str] = None
    code_artifacts: Optional[dict] = None
    force_upload: bool = False


# --- Shared Preparation Logic (Backend Agnostic) ---

def get_requirements():
    with open(config.SERVE_REQUIREMENTS_PATH, "r") as f:
        serve_reqs = f.read().splitlines()
        return serve_reqs

def get_existing_code_files():
    try:
        git_root = subprocess.check_output(
            "git rev-parse --show-toplevel", 
            shell=True, 
            text=True
        ).strip()
        git_root_path = Path(git_root).resolve()
    except subprocess.CalledProcessError:
        print("⚠️ Not a git repository! Falling back to current directory.")
        git_root_path = Path.cwd().resolve()

    git_output = subprocess.check_output(
        "git ls-files --cached --others --exclude-standard", 
        cwd=git_root_path,
        shell=True, 
        text=True
    )
        
    absolute_files = []
    for f in git_output.splitlines():
        full_path = (git_root_path / f).resolve()
        if full_path.exists():
            absolute_files.append(full_path)
            
    return absolute_files, git_root_path

def zip_code(artifacts_path: Path):
    # Change the extension to .zip
    code_path = artifacts_path / "code.zip"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    
    files, git_root = get_existing_code_files()
    
    if not files:
        print("⚠️ No valid files found to zip.")
        return

    print(f"📦 Creating archive at {code_path}...")
    
    # Open with zipfile, using ZIP_DEFLATED to actually compress the data
    with zipfile.ZipFile(code_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files:
            # Cut the absolute prefix exactly as before
            relative_arcname = file_path.relative_to(git_root)
            
            # Use .write() instead of .add()
            zipf.write(file_path, arcname=relative_arcname)
            
    print(f"✅ Zip created successfully at {code_path}")


def add_metadata(artifacts_path: Path, params: BaseParams, experiment_name: str, metrics: dict, empty_files=False):
    config_file_path = artifacts_path
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
    add_metadata(artifacts_path / config.PARAMS_SAVE_FILENAME, payload.params, experiment_name, payload.metrics)
    
    # 2. Save PyTorch Model
    # if isinstance(payload.tracker.logger, LitLogger):
    
    # 3. Copy Checkpoint if it exists
    if payload.ckpt_path: 
        shutil.copy(payload.ckpt_path, artifacts_path / payload.ckpt_path.split('/')[-1])
        
    # 4. Handle Code Artifacts (Zip Git or copy specific files)
    if payload.force_upload:
        return
    if payload.code_artifacts is None:
        zip_code(artifacts_path)
        return
        
    for artifact_name, source_path in payload.code_artifacts.items():
        source_path = Path(source_path)
        destination = artifacts_path / artifact_name
        if source_path.is_dir():
            shutil.copytree(source_path, destination, dirs_exist_ok=True) 
        else:
            shutil.copy(source_path, destination)



def is_metric_better_than_history(payload: ModelPayload) -> bool:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiment = client.get_experiment_by_name(payload.tracker.experiment)
        
        if not experiment:
            return True
        metric_name = f"best_{payload.monitor_metric}"
        mode = payload.monitor_mode
        filter_query = f"metrics.{metric_name} >= 0 AND attributes.run_id != '{payload.tracker.logger.run_id}'"

        order_direction = "ASC" if mode == "min" else "DESC"
        best_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_query, # Ensure the metric was actually logged
            order_by=[f"metrics.{metric_name} {order_direction}"],
            max_results=1
        )
        
        if not best_runs or metric_name not in best_runs[0].data.metrics:
            return True
            
        best_historical_value = best_runs[0].data.metrics[metric_name]
        current_value: float = payload.metrics.get(metric_name, 1)
        print(f"Comparing Current: {current_value:.4f} vs Historical Best: {best_historical_value:.4f}")
        
        if mode == "min":
            return current_value < best_historical_value
        else:
            return current_value > best_historical_value