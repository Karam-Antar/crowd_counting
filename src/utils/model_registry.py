"""Model registry functions for Lit and MLFlow integration."""

import shutil
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import torch
import urllib.parse

from src import config
from src.core.exported_model import ExportedModel
from src.core.params import BaseParams
from src.models.lit_model import BaseLitModel
from src.utils import helpers
from src.utils.experiment_trackers import BaseTracker
from src.utils.model_persistence import load_model, prepare_model_to_export, save_model


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
    
    # 3. Copy Checkpoint if it exists
    if payload.ckpt_path: 
        shutil.copy(payload.ckpt_path, artifacts_path / payload.ckpt_path.split('/')[-1])
        
    # 4. Handle Code Artifacts (Zip Git or copy specific files)
    if payload.code_artifacts is None:
        code_path = Path(f'{artifacts_path}/code.zip')
        code_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(f"git archive HEAD -o {code_path}", shell=True, check=True)
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


# --- The Registry Interface ---

class BaseRegistry(ABC):
    """Abstract interface for model registry strategies."""
    
    def __init__(self, experiment_name: Optional[str] = None, tracker: Any = None):
        self.experiment_name = experiment_name
        self.tracker = tracker

    # @abstractmethod
    # def model_exists(self, model_name: str) -> bool:
    #     """Checks if a model already exists in the registry to prevent duplicate runs."""
    #     pass

    @abstractmethod
    def upload_model(self, payload: ModelPayload):
        """Uploads the formatted payload directory to the specific artifact store."""
        pass

    @abstractmethod
    def download_model(self, model_name: str) -> tuple[list[str], str]:
        """Downloads the model directory and returns (list_of_relative_paths, local_download_dir)."""
        pass

    # --- Shared Loading Methods ---
    
    def load_model_from_ckpt(self, model_name: str, model_cls: type[BaseLitModel]):
        downloaded_paths, download_dir = self.download_model(model_name)
        ckpt_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.ckpt')), None)

        if not ckpt_relative_path:
            raise FileNotFoundError("No .ckpt file found in the downloaded model artifacts!")
        return model_cls.load_from_checkpoint(f'{download_dir}/{ckpt_relative_path}', weights_only=False)

    def load_params(self, downloaded_paths: list[str], download_dir: str):
        params_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.json')), None)
        if not params_relative_path:
            raise FileNotFoundError("No .json file found in the downloaded model artifacts!")
            
        # Let the exception raise naturally if JSON parsing fails to avoid returning None
        return BaseParams.from_json(f'{download_dir}/{params_relative_path}')

    def load_model(self, model_name: str, version: str = 'latest'):
        downloaded_paths, download_dir = self.download_model(model_name)
        model_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.pt2')), None)
        if not model_relative_path:
            raise FileNotFoundError("No .pt2 file found in the downloaded model artifacts!")
        
        model = ExportedModel(load_model(f'{download_dir}/{model_relative_path}'))
        params = self.load_params(downloaded_paths, download_dir)
        return model, params


# --- Concrete Implementations ---

class LitRegistry(BaseRegistry):

    def upload_model(self, payload: ModelPayload):
        import litmodels
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_path = Path(tmp_dir)
            prepare_temp_dir(artifacts_path, payload, payload.tracker.full_experiment_name)
            
            lit_exp = self.tracker.active_experiment 
            model_url = f'{lit_exp.teamspace.owner.name}/{lit_exp.teamspace.name}/{payload.tracker.model_name}'
            model_info = litmodels.upload_model_files(name=model_url, path=artifacts_path)
            lit_exp.log_metadata({'model_registry': f'{model_url}:{model_info.version}'})

    def download_model(self, model_name: str) -> tuple[list[str], str]:
        import litmodels
        model_suffix = model_name.rsplit('/', 1)[-1]
        download_dir = f"{config.MODEL_SAVE_PATH}/{model_suffix}"
        downloaded_paths = litmodels.download_model(model_name, download_dir=download_dir)
        return downloaded_paths, download_dir


class MLFlowRegistry(BaseRegistry):

    def log_model(self, payload: ModelPayload):
        import mlflow
        from mlflow.models import infer_signature
        input, dynamic_shapes = prepare_model_to_export(payload.params, payload.model)
        with torch.no_grad():
            predicted_map = payload.model(input)
        signature = infer_signature(input.numpy(), predicted_map.numpy())
        return mlflow.pytorch.log_model(
            pytorch_model=payload.model,
            artifact_path=payload.tracker.model_name,
            serialization_format="pt2",
            input_example=input.numpy(),
            signature=signature,
            dynamic_shapes=dynamic_shapes,
        )


    def upload_model(self, payload: ModelPayload):
        import mlflow
        with mlflow.start_run(run_id=payload.tracker.logger.run_id, nested=True):
            
            model_info = self.log_model(payload)  
            # 2. Register the model
            mlflow.register_model(model_uri=model_info.model_uri, name=payload.tracker.experiment)

            with tempfile.TemporaryDirectory() as tmp_dir:
                artifacts_path = Path(tmp_dir)
            
            # 3. Add custom metadata and files
                # The signature here is now clean and beautiful!
                prepare_temp_dir(artifacts_path, payload, payload.tracker.full_experiment_name)
                
                mlflow.log_artifacts(local_dir=str(artifacts_path), artifact_path=payload.tracker.model_name)

    def download_model(self, model_name: str, version: str = "latest") -> tuple[list[str], str]:
        import mlflow
        
        safe_name = model_name.replace("/", "_")
        download_dir = f"{config.MODEL_SAVE_PATH}/{safe_name}"
        
        model_uri = f"models:/{model_name}/{version}"
        
        try:
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_dir)
        except Exception as e:
            print(f"Failed to pull from registry via {model_uri}. Attempting direct artifact pull...")
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_name, dst_path=download_dir)
        
        downloaded_paths = []
        for root, _, files in os.walk(local_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), local_path)
                downloaded_paths.append(rel_path)
                
        return downloaded_paths, local_path

# """Model registry functions for Lit (LitModels) integration."""

# import shutil
# import os
# import subprocess
# import tempfile
# from pathlib import Path
# import litmodels
# from lightning.pytorch import LightningModule
# import torch
# from src import config
# from src.core.exported_model import ExportedModel
# from src.core.params import BaseParams
# from src.models.lit_model import BaseLitModel
# from src.utils import helpers
# from src.utils.model_persistence import load_model, save_model
# import urllib.parse
# from litlogger import Experiment
# # import json
# # from src.utils.registries import PARAMS_REGISTRY


# def add_metadata(artifacts_path: Path, params: BaseParams, lit_experiment: Experiment):
#     config_file_path = artifacts_path / "model_config.json"
#     metrics: dict = {k: v[0] for k, v in lit_experiment.metrics.items() if str(k).casefold().startswith('best')}
#     params.to_json(config_file_path, meta={'experiment': lit_experiment.name, 'metrics': metrics})
#     helpers.create_empty_text_files(artifacts_path, [f'experiment={urllib.parse.quote(lit_experiment.name, safe='=')}', *helpers.generate_file_names(metrics)])


# def prepare_temp_dir(artifacts_path: Path, model: torch.nn.Module, params: BaseParams, lit_experiment: Experiment, code_artifacts: dict | None = None, ckpt_path: str | None = None):
#     os.makedirs(artifacts_path, exist_ok=True)
#         # Copy and rename to avoid collisions
#     add_metadata(artifacts_path, params, lit_experiment)
#     save_model(model, params, model_id='model', model_path=artifacts_path)
#     if ckpt_path: 
#         shutil.copy(ckpt_path, artifacts_path / ckpt_path.split('/')[-1])   # Keep original name
#     if code_artifacts is None:
#         code_path = Path(f'{artifacts_path}/code.zip')
#         code_path.parent.mkdir(parents=True, exist_ok=True)
#         cmd = f"git archive HEAD -o{code_path}"
#         try:
#             subprocess.run(cmd, shell=True, check=True)
#             print(f"✅ Repository successfully zipped to {code_path}")
#         except subprocess.CalledProcessError as e:
#             print(f"❌ Failed to create zip: {e}")
#         return
#     for artifact_name, source_path in code_artifacts.items():
#         source_path = Path(source_path)
#         destination = artifacts_path / artifact_name
        
#         if source_path.is_dir():
#             # Use copytree for folders (dirs_exist_ok=True prevents errors if it exists)
#             shutil.copytree(source_path, destination, dirs_exist_ok=True)
#         else:
#             # Use copy for individual files
#             shutil.copy(source_path, destination)



# def upload_model_to_lit(model: torch.nn.Module, model_name: str, lit_experiment: Experiment, params: BaseParams, code_artifacts: dict | None = None, ckpt_path: str | None = None):
#     """Upload the model file to Lit and log it as an artifact."""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         artifacts_path = Path(tmp_dir)
#         # Create a temporary folder to organize files
#         prepare_temp_dir(artifacts_path, model, params, lit_experiment, code_artifacts, ckpt_path)
#         model_url = f'{lit_experiment.teamspace.owner.name}/{lit_experiment.teamspace.name}/{model_name}'
#         model_info = litmodels.upload_model_files(name=model_url, path=artifacts_path)
#         lit_experiment.log_metadata({'model_registry': f'{model_url}:{model_info.version}'})


# def download_model_from_lit(model_name: str):
#     model_suffix = model_name.rsplit('/', 1)[-1]
#     print(model_suffix)  # Get the last part of the path as the model name
#     download_dir = f"{config.MODEL_SAVE_PATH}/{model_suffix}"
#     downloaded_paths = litmodels.download_model(
#         model_name,
#         download_dir=download_dir,
#     )
#     return downloaded_paths, download_dir


# def load_model_from_lit_ckpt(model_name: str, model_cls: type[BaseLitModel]):
#     """Download the model files from Lit and load the model."""
#     downloaded_paths, download_dir = download_model_from_lit(model_name)
#     ckpt_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.ckpt')), None)

#     if ckpt_relative_path is None:
#         raise FileNotFoundError("No .ckpt file found in the downloaded model artifacts!")
#     model = model_cls.load_from_checkpoint(
#         f'{download_dir}/{ckpt_relative_path}', weights_only=False
#     )
#     return model

# def load_params(downloaded_paths: list[str], download_dir: str):
#     params_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.json')), None)
#     if params_relative_path is None:
#         raise FileNotFoundError("No .json file found in the downloaded model artifacts!")
#     try:
#         # with open(f'{download_dir}/{params_relative_path}', 'r') as file:
#         #     data = json.load(file)
#         # flat_data = helpers.flatten_dict(data, to_str=True)
#         # params = PARAMS_REGISTRY[flat_data.get('params.model_class')].from_dict(data)
#         params = BaseParams.from_json(f'{download_dir}/{params_relative_path}')
#     except Exception as e:
#         print(e)
#         params = None
#     return params


# def load_model_from_lit(model_name):
#     downloaded_paths, download_dir = download_model_from_lit(model_name)
#     model_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.pt2')), None)
#     if model_relative_path is None:
#         raise FileNotFoundError("No .pt2 file found in the downloaded model artifacts!")
#     model = ExportedModel(load_model(f'{download_dir}/{model_relative_path}'))
#     params = load_params(downloaded_paths, download_dir)
#     return model, params