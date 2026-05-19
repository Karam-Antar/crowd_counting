import tempfile
from pathlib import Path
import torch
import mlflow
from mlflow.models import infer_signature
from unittest.mock import patch
from src import config
from src.utils.model_persistence import prepare_model_to_export, save_model
from src.utils.model_registry.base import BaseRegistry
from src.utils.model_registry.utils import ModelPayload, prepare_temp_dir

class MLFlowRegistry(BaseRegistry):


    def log_model(self, payload: ModelPayload):
        input_sample = torch.randn(1, 3, 256, 256)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model, _ = save_model(payload.model, payload.params, model_id='model', model_path=Path(tmp_dir))
        with torch.no_grad():
            predicted_map = payload.model(input_sample)
        input_sample = input_sample.numpy().copy()
        signature = infer_signature(input_sample, predicted_map.numpy())
        with patch("torch.export.export", return_value=model):
            return mlflow.pytorch.log_model(
                pytorch_model=payload.model,
                name=payload.tracker.model_name,
                serialization_format="pt2",
                input_example=input_sample,
                signature=signature,
                # dynamic_shapes=dynamic_shapes,
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



# from mlflow.pyfunc import PythonModel

# class DynamicPT2Wrapper(PythonModel):
#     def load_context(self, context):
#         import torch
#         # Load the custom .pt2 file that we packaged into the artifacts
#         self.model = torch.export.load(context.artifacts["model_file"])
        
#     def predict(self, context, model_input):
#         import torch
#         from src.core import inference
#         import numpy as np
        
#         # 1. Extract the raw numpy array safely
#         if hasattr(model_input, "values"):
#             # It's a Pandas DataFrame
#             data = model_input.values
#         elif isinstance(model_input, dict):
#             # It's a dictionary of inputs, grab the first one
#             data = next(iter(model_input.values()))
#         else:
#             # It's already a numpy array
#             data = model_input 
            
#         # 2. Convert to tensor (using .copy() to avoid the "non-writable buffer" warning)
#         input_tensor = torch.tensor(np.array(data).copy())
#         return inference.predict(self.model.module(), input_tensor)


# class MLFlowRegistry(BaseRegistry):

#     def log_model(self, payload: ModelPayload):
#         # Rename to input_tensor to avoid shadowing Python's built-in input()
#         input_tensor = torch.randn(1, 3, 256, 256)
        
#         with torch.no_grad():
#             predicted_map = payload.model(input_tensor)
            
#         input_numpy = input_tensor.numpy().copy()
#         signature = infer_signature(input_numpy, predicted_map.numpy())
        
#         # 1. Open the temp directory
#         with tempfile.TemporaryDirectory() as tmp_dir:
            
#             # 2. Unpack the tuple returned by your save_model function
#             # (Assuming save_model returns (ExportedProgram, path_string))
#             _, model_path = save_model(
#                 payload.model, 
#                 payload.params, 
#                 model_id='model', 
#                 model_path=Path(tmp_dir)
#             )
            
#             # 3. Convert the exact file path into a file:// URI for MLflow
#             # pt2_uri = pathlib.Path(actual_path).resolve().as_uri()
            
#             # 4. LOG INSIDE THE WITH BLOCK! 
#             # If you un-indent this, the file gets deleted before MLflow can read it.
#             return mlflow.pyfunc.log_model(
#                 artifact_path=payload.tracker.model_name, # Usually artifact_path is preferred over name here
#                 python_model=DynamicPT2Wrapper(),
#                 artifacts={"model_file": str(model_path)},  # Use the clean URI
#                 signature=signature,
#                 input_example=input_numpy,
#             )