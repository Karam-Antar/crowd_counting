import tempfile
from pathlib import Path
import numpy as np
import torch
import mlflow
from unittest.mock import patch
from src import config
from src.core.model_wrapper import ModelWrapper
from src.core.pyfunc import ProductionPyTorchWrapper
from src.utils.model_registry.base import BaseRegistry
from src.utils.model_registry.utils import ModelPayload, add_metadata, get_existing_code_files, get_requirements, is_metric_better_than_history, prepare_temp_dir, zip_code
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


class MLFlowRegistry(BaseRegistry):

    def load_model(self, model_uri: str, **kwargs):
        """
        Loads an MLflow pyfunc model directly via its URI and unwraps it.
        Example model_uri: "models:/crowd_counting/49" or "models:/crowd_counting/@champion"
        """
        # 1. Download and load the PyFunc wrapper into memory automatically
        pyfunc_wrapper = mlflow.pyfunc.load_model(model_uri)
        
        # 2. Unwrap it to expose your custom ProductionPyTorchWrapper instance
        custom_class_instance = pyfunc_wrapper.unwrap_python_model()
        
        # 3. Extract the fully initialized PyTorch model and parameters
        # (These were automatically populated by your load_context method)
        model = ModelWrapper(custom_class_instance.model)
        params = custom_class_instance.params
        
        return model, params


    def log_model(self, payload: ModelPayload):
        input_schema = Schema([
            TensorSpec(np.dtype(np.uint8), (-1, -1, -1, 3))
        ])

        # 2. Create the Signature
        # By omitting the output schema (or explicitly setting it to None), 
        # we allow the density_map to scale dynamically alongside the input image.
        signature = ModelSignature(inputs=input_schema, outputs=None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir)
            weights_path = save_path / "weights.pt"
            params_path = save_path / config.PARAMS_SAVE_FILENAME
            torch.save(payload.model.model.state_dict(), weights_path)
            add_metadata(params_path, payload.params, payload.tracker.full_experiment_name, payload.metrics)
            return mlflow.pyfunc.log_model(
                name=payload.tracker.model_name,
                python_model=config.PYFUNC_MODEL_PATH,
                artifacts={
                    "weights": str(weights_path),
                    "params": str(params_path)
                },
                code_paths=[config.SERVE_CODE_PATH], 
                signature=signature, # Use the new multi-image signature
                pip_requirements=get_requirements(),
            )


    def upload_model(self, payload: ModelPayload):
        if not payload.force_upload and not is_metric_better_than_history(payload):
            print(f"Model did not outperform historical best. Skipping upload to registry.")
            with mlflow.start_run(run_id=payload.tracker.logger.run_id, nested=True):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    artifacts_path = Path(tmp_dir)
                    zip_code(artifacts_path)  # Ensure code is still logged for traceability
                    mlflow.log_artifacts(local_dir=str(artifacts_path), artifact_path=payload.tracker.model_name)
            return
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

    def download_model(self, model_name: str, version: str = "latest", run_id: str | None = None) -> tuple[list[str], str]:
        import os
        
        safe_name = model_name.replace("/", "_")
        download_dir = f"{config.MODEL_SAVE_PATH}/{safe_name}:{version}"
        
        model_uri = f"models:/{model_name}/{version}"
        run_uri = f'runs:/{run_id}/' if run_id else None
        
        try:
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_dir)
        except Exception as e:
            print(f"Failed to pull from registry via {model_uri}")
        if run_uri:
            try:
                mlflow.artifacts.download_artifacts(artifact_uri=run_uri, dst_path=download_dir)
            except Exception as e:
                print(f"Failed to pull from run via {run_uri}")
        downloaded_paths = []
        for root, _, files in os.walk(local_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), local_path)
                downloaded_paths.append(rel_path)
                
        return downloaded_paths, local_path
