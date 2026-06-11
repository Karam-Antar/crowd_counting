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

def get_existing_code_files():
    # 1. Get the list of files from Git
    git_output = subprocess.check_output(
            "git ls-files --cached --others --exclude-standard", 
            shell=True, 
            text=True
        )
        
    # 2. Filter out files that have been deleted locally
    existing_files = [f for f in git_output.splitlines() if Path(f).exists()]
    return existing_files


def zip_code(artifacts_path: Path):
    code_path = artifacts_path / "code.tar.gz"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        existing_files = get_existing_code_files()
        
        if not existing_files:
            print("⚠️ No valid files found to zip.")
            return

        # 3. Pass the filtered list to tar via standard input
        subprocess.run(
            f"tar -czvf '{code_path.as_posix()}' -T -", 
            input="\n".join(existing_files),
            shell=True, 
            check=True,
            text=True
        )
        print(f"✅ Zip created successfully at {code_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create zip: {e}")


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