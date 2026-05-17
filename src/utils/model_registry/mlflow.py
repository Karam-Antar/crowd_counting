import tempfile
from pathlib import Path
import torch
import mlflow
from mlflow.models import infer_signature

from src import config
from src.utils.model_persistence import prepare_model_to_export
from src.utils.model_registry.base import BaseRegistry
from src.utils.model_registry.utils import ModelPayload, prepare_temp_dir


class MLFlowRegistry(BaseRegistry):

    def log_model(self, payload: ModelPayload):
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
        import os
        
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
