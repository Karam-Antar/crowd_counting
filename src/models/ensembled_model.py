from typing import Dict, List, Optional, Union
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MetricCollection

class EnsembledCrowdCounter:
    def __init__(self, models: list, metrics: Optional[MetricCollection] = None):
        """
        Args:
            models (list): A list of pre-loaded ProductionPyTorchWrapper objects.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not models:
            raise ValueError("You must provide at least one model wrapper to initialize the ensemble.")
        self.models = models
        
        if metrics is None:
            # Standard Crowd Counting metrics
            self.metrics = MetricCollection({
                "MAE": MeanAbsoluteError(),
                "RMSE": MeanSquaredError(squared=False),
                "NAE": MeanAbsolutePercentageError(),
            }).to(self.device)
        else:
            self.metrics = metrics.to(self.device)

    def predict(self, model_input: np.ndarray) -> List[Dict[str, Union[float, np.ndarray]]]:
        """
        Aggregates outputs from all sub-models and computes the soft-voted (averaged) result.
        Uses vectorized NumPy operations to eliminate manual loop accumulation.
        """
        # 1. Collect predictions from all underlying wrappers
        # all_model_outputs shape: (num_models, batch_size) list of dicts
        all_model_outputs = [model.predict(model_input) for model in self.models]

        # 2. Extract counts and density maps into NumPy arrays for vectorization
        counts = np.array([[pred["count"] for pred in batch] for batch in all_model_outputs])
        density_maps = np.array([[pred["density_map"] for pred in batch] for batch in all_model_outputs])

        # 3. Perform soft voting (arithmetic mean) across the model dimension (axis 0)
        avg_counts = counts.mean(axis=0)
        avg_density_maps = density_maps.mean(axis=0)

        # 4. Pack back into the exact original dictionary structure
        return [
            {"count": float(c), "density_map": d} 
            for c, d in zip(avg_counts, avg_density_maps)
        ]

    def evaluate(self, data: Union[DataLoader, pl.LightningDataModule]) -> Dict[str, float]:
        """
        Runs evaluation by loading raw images from paths provided by the DataLoader.
        """
        if isinstance(data, pl.LightningDataModule):
            data.setup(stage="test")
            loader = data.test_dataloader() or data.val_dataloader()
        else:
            loader = data
        
        self.metrics.reset()

        # The loader now yields a tuple of image paths (strings) and the ground truth tensor
        for batch_paths, batch_y in loader:
            batch_y = batch_y.to(self.device)
            
            # 1. Load raw images directly from the disk into a list of NumPy arrays
            # PIL is safer here than cv2 because it natively opens in RGB format.
            raw_images = np.array([np.array(Image.open(path).convert("RGB")) for path in batch_paths])
            
            # 2. Forward pass
            # NOTE: If your images are varying sizes, `predict` must be updated to 
            # accept a List[np.ndarray] instead of a single stacked np.ndarray.
            pred = self.predict(raw_images)
            
            # 3. Calculate the counts
            pred_counts_list = [img_pred['count'] for img_pred in pred]
            pred_count_tensor = torch.tensor(pred_counts_list, dtype=torch.float32, device=self.device)
            
            # Sum spatial/channel dimensions for ground truth (if GT is a density map)
            # If your new dataloader just returns the integer count, change this to `gt_count = batch_y`
            if batch_y.ndim > 1:
                gt_count = batch_y.sum(dim=(1, 2, 3))
            else:
                gt_count = batch_y
            
            # 4. Update metrics
            self.metrics.update(pred_count_tensor, gt_count)

        results = self.metrics.compute()
        return {name: val.item() for name, val in results.items()}