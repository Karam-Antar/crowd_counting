import mlflow.pyfunc
import torch
import json
import numpy as np

from src.core import inference

class ProductionPyTorchWrapper(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for loading and serving a crowd-counting model.

    The wrapper resolves model artifacts at runtime, restores the trained state
    dictionary, and exposes a predict method that consumes raw NumPy image arrays and
    returns crowd counts plus decoded density maps.
    """
    
    def load_context(self, context):
        """Load model configuration, weights, and preprocessing metadata from MLflow.

        Args:
            context: MLflow model context containing the artifact directory and metadata.

        Returns:
            None: The model and its parameters are loaded onto the instance in place.

        Raises:
            KeyError: If the expected ``weights`` or ``params`` artifacts are missing.
            RuntimeError: If the checkpoint cannot be mapped to the detected device.
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
        """Serve a batch of raw images through the loaded crowd-counting model.

        Args:
            context: MLflow model context, used for runtime access to the loaded model.
            model_input: A NumPy array representing one or more RGB images, typically with
                shape ``(H, W, C)`` or ``(B, H, W, C)`` and uint8 values.

        Returns:
            list[dict[str, np.ndarray]]: A list of result dictionaries, each containing a
            floating-point crowd count and a density map array for one image.

        Raises:
            ValueError: If the supplied model input cannot be converted to a valid image
                tensor layout.
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