import torch
from torch.fx import GraphModule
import torch.nn.functional as F
from torch.export import ExportedProgram
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torchmetrics
from torchmetrics import MeanAbsolutePercentageError, MetricCollection, MeanAbsoluteError, MeanSquaredError
from typing import Union, Dict, Optional, Tuple
from src import config
from src.core import inference

class ExportedModel:
    """
    A production wrapper for torch.ExportedProgram tailored for Crowd Counting.
    Handles high-level evaluation and single-input inference.
    """
    def __init__(self, exported_program: ExportedProgram | GraphModule, device: str = config.device, metrics: Optional[MetricCollection] = None):
        self.exported_program = exported_program if isinstance(exported_program, ExportedProgram) else None
        self.device = torch.device(device)
        
        # Extract the optimized callable module from the exported program
        self.model = (self.exported_program.module() if self.exported_program else exported_program).to(device)
        
        if metrics is None:
            # Standard Crowd Counting metrics
            self.metrics = MetricCollection({
                "MAE": MeanAbsoluteError(),
                "RMSE": MeanSquaredError(squared=False),  # RMSE
                'NAE': MeanAbsolutePercentageError(),
            }).to(self.device)
        else:
            self.metrics = metrics.to(self.device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Custom inference method for single inputs.
        Returns:
            - total_count (float): The estimated number of people in the image.
            - density_map (torch.Tensor): The 2D spatial distribution of the crowd.
        """
        return inference.predict(self.model, x)
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        return self.model(x)

    @torch.no_grad()
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
            pred_density = self.model(batch_x)
            # pred_density = F.relu(pred_density)
            
            # 2. Calculate the counts by summing across spatial and channel dimensions
            # Assuming shape is [Batch, Channel, Height, Width]
            pred_count = pred_density.sum(dim=(1, 2, 3)) / config.LABEL_SCALER
            gt_count = batch_y.sum(dim=(1, 2, 3)) / config.LABEL_SCALER
            print(pred_count, gt_count)
            
            # 3. MetricCollection updates all metrics simultaneously
            self.metrics.update(pred_count, gt_count)

        # compute() returns a dict: {'MAE': tensor(12.5), 'MSE': tensor(150.2)}
        results = self.metrics.compute()
        
        # Convert tensors to standard python floats for the final return
        return {name: val.item() for name, val in results.items()}