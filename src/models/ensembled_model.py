from typing import Dict, Optional, Union
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MetricCollection
from src import config

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
                "RMSE": MeanSquaredError(squared=False),  # RMSE
                'NAE': MeanAbsolutePercentageError(),
            }).to(self.device)
        else:
            self.metrics = metrics.to(self.device)

    def predict(self, model_input: np.ndarray) -> list[dict[str, np.ndarray]]:
        """
        Aggregates outputs from all sub-models and computes the soft-voted (averaged) result.
        Preserves the exact signature and return format of ProductionPyTorchWrapper.
        """
        # 1. Collect predictions from all underlying wrappers
        # Each model.predict returns a list[dict[str, np.ndarray]]
        all_model_outputs = [model.predict(model_input) for model in self.models]

        num_models = len(self.models)
        batch_size = len(all_model_outputs[0])
        ensemble_results = []

        # 2. Iterate image by image through the batch to compute averages
        for img_idx in range(batch_size):
            total_count = 0.0
            total_density_map = None

            for model_idx in range(num_models):
                single_pred = all_model_outputs[model_idx][img_idx]
                
                total_count += single_pred["count"]
                
                # Initialize the numpy array accumulator on the first model iteration
                if total_density_map is None:
                    total_density_map = np.copy(single_pred["density_map"])
                else:
                    total_density_map += single_pred["density_map"]

            # 3. Perform soft voting (arithmetic mean)
            avg_count = total_count / num_models
            avg_density_map = total_density_map / num_models

            # 4. Pack back into the exact original dictionary structure
            ensemble_results.append({
                "count": float(avg_count),
                "density_map": avg_density_map
            })

        return ensemble_results
    

    def evaluate(self, data: Union[DataLoader, pl.LightningDataModule]) -> Dict[str, float]:
        """
        Runs evaluation and returns a dictionary of all computed metrics (e.g., MAE, MSE).
        """
        if isinstance(data, pl.LightningDataModule):
            data.setup(stage="test")
            loader = data.test_dataloader() or data.val_dataloader()
        else:
            loader = data
        
        # Reset metrics to ensure a clean slate for this run
        self.metrics.reset()

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # 1. Forward pass
            pred = self.predict(batch_x.permute(0,2,3,1).cpu().numpy().astype('uint8'))
            
            # 2. Calculate the counts by summing across spatial and channel dimensions
            # Assuming shape is [Batch, Channel, Height, Width]
            pred_counts_list = [img_pred['count'] for img_pred in pred]
            
            # 3. FIX: Convert predictions to a PyTorch tensor on the correct GPU/CPU device
            pred_count_tensor = torch.tensor(pred_counts_list, dtype=torch.float32, device=self.device)
            gt_count = batch_y.sum(dim=(1, 2, 3))
            print(pred_count_tensor, gt_count)
            
            # 3. MetricCollection updates all metrics simultaneously
            self.metrics.update(pred_count_tensor, gt_count)

        # compute() returns a dict: {'MAE': tensor(12.5), 'MSE': tensor(150.2)}
        results = self.metrics.compute()
        
        # Convert tensors to standard python floats for the final return
        return {name: val.item() for name, val in results.items()}