import mlflow.pyfunc
import torch
import json
import numpy as np

from src.core import inference
from src.data.transform import UnpadToOriginal

class ProductionPyTorchWrapper(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        """
        Executed exactly once when the container boots up.
        Handles loading code, parameters, and weights safely.
        """
        # 1. Determine hardware environment safely
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Delayed imports (resolved via MLflow's code_paths)
        from src.models.model import CrowdCounter
        from src.core.params import BaseParams
        self.params = BaseParams.from_json(context.artifacts['params'])

        # 4. Initialize the empty architecture using your parameters
        self.model = CrowdCounter(params=self.params)

        # 5. Safely load weights and map them to the correct device
        state_dict_path = context.artifacts["weights"]
        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # 6. Lock the model for inference
        self.model.eval()
        self.model.to(self.device)

    def predict(self, context, model_input) -> list[dict[str, np.ndarray]]:
        """
        Executed for every REST API request.
        Handles I/O translation and preprocessing.
        """
        # 1. Delayed import for preprocessing logic
        from src.data.transform_sample import preprocess

        # 2. Apply your custom preprocessing using the loaded params
        # (model_input is a NumPy array parsed automatically by MLflow)
        # input_tensor = torch.tensor(model_input, dtype=torch.float32).to(self.device)
        model_input = model_input.astype('uint8')
        raw_tensor = torch.tensor(model_input)
        
        # 2. Safety check: If a client happens to send a single unbatched image [H, W, C], 
        # add the batch dimension dynamically so it becomes [1, H, W, C]
        if raw_tensor.ndim == 3:
            raw_tensor = raw_tensor.unsqueeze(0)
            
        # 3. Permute the entire batch: [B, H, W, C] -> [B, C, H, W]
        raw_tensor = raw_tensor.permute(0, 3, 1, 2)
        h, w = raw_tensor.shape[-2:]
        img = preprocess(raw_tensor, self.params)
        if isinstance(img, list) and len(img) == 1:
            img = img[0]
            
        # 2. If preprocess returned a list of multiple individual tensors, batch them:
        elif isinstance(img, list):
            img = torch.stack(img)
            
        # 3. If it returned a NumPy array instead of a Tensor, convert it:
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        # 3. Convert preprocessed NumPy array to a PyTorch Tensor

        # 4. Run inference safely
        # Model returns two outputs: a float/vector score and a matrix/tensor density map
        counts, density_maps = inference.predict(self.model, img, device=self.device, h=h, w=w)
        # 5. Post-processing: Move off GPU and convert back to standard NumPy arrays
        counts = counts.cpu().numpy()
        density_maps = density_maps.cpu().numpy()

        # 6. Return as a dictionary for clean REST JSON formatting
        results = []
        for i in range(len(counts)):
            results.append({
                "count": float(counts[i]), 
                # Using maps[i] correctly extracts the [1, H, W] map for this specific image
                "density_map": density_maps[i] 
            })

        return results


mlflow.models.set_model(ProductionPyTorchWrapper())