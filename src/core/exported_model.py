import torch
from torch.export import ExportedProgram
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from typing import Union, List, Dict, Optional, Tuple
from src import config
from src.core import inference

class ExportedModel:
    """
    A production wrapper for torch.ExportedProgram.
    Handles high-level evaluation and single-input inference.
    """
    def __init__(self, exported_program: ExportedProgram, class_names: List[str]=config.CLASS_NAMES, device: str = config.device, metrics: Optional[MetricCollection] = None):
        self.exported_program = exported_program
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.device = torch.device(device)
        
        # Extract the optimized callable module from the exported program
        self.model = self.exported_program.module().to(self.device)
        # self.model.eval()
        if metrics is None:
            # Default suite of metrics if none provided
            self.metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=self.num_classes),
                "precision": Precision(task="multiclass", num_classes=self.num_classes),
                "recall": Recall(task="multiclass", num_classes=self.num_classes),
                "f1": F1Score(task="multiclass", num_classes=self.num_classes)
            }).to(self.device)
        else:
            self.metrics = metrics.to(self.device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[str, int, float]:
        """
        Custom inference method for single inputs.
        """
        return inference.predict(self.model, x, self.device)
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        return self.model(x)

    @torch.no_grad()
    def evaluate(self, data: Union[DataLoader, pl.LightningDataModule]) -> Dict[str, float]:
        """
        Runs evaluation and returns a dictionary of all computed metrics.
        """
        if isinstance(data, pl.LightningDataModule):
            data.setup()
            loader = data.val_dataloader() or data.test_dataloader()
        else:
            loader = data
        
        # Reset metrics to ensure a clean slate for this run
        self.metrics.reset()

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            logits = self.model(batch_x)
            preds = torch.argmax(logits, dim=1)
            
            # MetricCollection updates all metrics at once
            self.metrics.update(preds, batch_y)

        # compute() returns a dict: {'acc': tensor(0.9), 'f1': tensor(0.88), ...}
        results = self.metrics.compute()
        
        # Convert tensors to standard python floats for the final return
        return {name: val.item() for name, val in results.items()}