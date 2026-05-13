"""Model registry functions for Lit (LitModels) integration."""

import shutil
import os
import subprocess
import tempfile
from pathlib import Path
import litmodels
from lightning.pytorch import LightningModule
import torch
from src import config
from src.core.exported_model import ExportedModel
from src.core.params import BaseParams
from src.models.lit_model import BaseLitModel
from src.utils import helpers
from src.utils.model_persistence import load_model, save_model
import urllib.parse
from litlogger import Experiment
# import json
# from src.utils.registries import PARAMS_REGISTRY


def add_metadata(artifacts_path: Path, params: BaseParams, lit_experiment: Experiment):
    config_file_path = artifacts_path / "model_config.json"
    metrics: dict = {k: v[0] for k, v in lit_experiment.metrics.items() if str(k).casefold().startswith('best')}
    params.to_json(config_file_path, meta={'experiment': lit_experiment.name, 'metrics': metrics})
    helpers.create_empty_text_files(artifacts_path, [f'experiment={urllib.parse.quote(lit_experiment.name, safe='=')}', *helpers.generate_file_names(metrics)])


def prepare_temp_dir(artifacts_path: Path, model: torch.nn.Module, params: BaseParams, lit_experiment: Experiment, code_artifacts: dict | None = None, ckpt_path: str | None = None):
    os.makedirs(artifacts_path, exist_ok=True)
        # Copy and rename to avoid collisions
    add_metadata(artifacts_path, params, lit_experiment)
    save_model(model, params, model_id='model', model_path=artifacts_path)
    if ckpt_path: 
        shutil.copy(ckpt_path, artifacts_path / ckpt_path.split('/')[-1])   # Keep original name
    if code_artifacts is None:
        code_path = Path(f'{artifacts_path}/code.zip')
        code_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = f"git archive HEAD -o{code_path}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ Repository successfully zipped to {code_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create zip: {e}")
        return
    for artifact_name, source_path in code_artifacts.items():
        source_path = Path(source_path)
        destination = artifacts_path / artifact_name
        
        if source_path.is_dir():
            # Use copytree for folders (dirs_exist_ok=True prevents errors if it exists)
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
        else:
            # Use copy for individual files
            shutil.copy(source_path, destination)



def upload_model_to_lit(model: torch.nn.Module, model_name: str, lit_experiment: Experiment, params: BaseParams, code_artifacts: dict | None = None, ckpt_path: str | None = None):
    """Upload the model file to Lit and log it as an artifact."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts_path = Path(tmp_dir)
        # Create a temporary folder to organize files
        prepare_temp_dir(artifacts_path, model, params, lit_experiment, code_artifacts, ckpt_path)
        model_url = f'{lit_experiment.teamspace.owner.name}/{lit_experiment.teamspace.name}/{model_name}'
        model_info = litmodels.upload_model_files(name=model_url, path=artifacts_path)
        lit_experiment.log_metadata({'model_registry': f'{model_url}:{model_info.version}'})


def download_model_from_lit(model_name: str):
    model_suffix = model_name.rsplit('/', 1)[-1]
    print(model_suffix)  # Get the last part of the path as the model name
    download_dir = f"{config.MODEL_SAVE_PATH}/{model_suffix}"
    downloaded_paths = litmodels.download_model(
        model_name,
        download_dir=download_dir,
    )
    return downloaded_paths, download_dir


def load_model_from_lit_ckpt(model_name: str, model_cls: type[BaseLitModel]):
    """Download the model files from Lit and load the model."""
    downloaded_paths, download_dir = download_model_from_lit(model_name)
    ckpt_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.ckpt')), None)

    if ckpt_relative_path is None:
        raise FileNotFoundError("No .ckpt file found in the downloaded model artifacts!")
    model = model_cls.load_from_checkpoint(
        f'{download_dir}/{ckpt_relative_path}', weights_only=False
    )
    return model

def load_params(downloaded_paths: list[str], download_dir: str):
    params_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.json')), None)
    if params_relative_path is None:
        raise FileNotFoundError("No .json file found in the downloaded model artifacts!")
    try:
        # with open(f'{download_dir}/{params_relative_path}', 'r') as file:
        #     data = json.load(file)
        # flat_data = helpers.flatten_dict(data, to_str=True)
        # params = PARAMS_REGISTRY[flat_data.get('params.model_class')].from_dict(data)
        params = BaseParams.from_json(f'{download_dir}/{params_relative_path}')
    except Exception as e:
        print(e)
        params = None
    return params


def load_model_from_lit(model_name):
    downloaded_paths, download_dir = download_model_from_lit(model_name)
    model_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.pt2')), None)
    if model_relative_path is None:
        raise FileNotFoundError("No .pt2 file found in the downloaded model artifacts!")
    model = ExportedModel(load_model(f'{download_dir}/{model_relative_path}'))
    params = load_params(downloaded_paths, download_dir)
    return model, params