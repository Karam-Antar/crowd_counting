import torch
from torch.fx import GraphModule
import torch.nn.functional as F
from torch.export import ExportedProgram
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torchmetrics
from torchmetrics import MetricCollection
from typing import Union, Dict, Optional, Tuple
from src import config
from src.core import inference
from src.data.datamodule import CrowdDataModule
from src.models.lit_model import BaseLitModel
from src.models.model import CrowdCounter
import lightning.pytorch as pl
from src.models.metrics import MeanBiasError, PositiveOnlyNAE

class ExportedModel:
    """
    A production wrapper for torch.ExportedProgram tailored for Crowd Counting.
    Handles high-level evaluation and single-input inference.
    """
    def __init__(self, exported_program: ExportedProgram | GraphModule, device: str = config.device, metrics: Optional[MetricCollection] = None):
        self.exported_program = exported_program if isinstance(exported_program, ExportedProgram) else None
        self.device = torch.device(device)
        
        # Extract the optimized callable module from the exported program
        self.model: CrowdCounter = (self.exported_program.module() if self.exported_program else exported_program).to(device)
        
        if metrics is None:
            # Standard Crowd Counting metrics
            self.metrics = torchmetrics.MetricCollection({
                'mae': torchmetrics.MeanAbsoluteError(),
                'rmse': torchmetrics.MeanSquaredError(squared=False),
                'nae': PositiveOnlyNAE(),
                'mbe': MeanBiasError(),
            })
            self.mask_metrics = torchmetrics.MetricCollection({
                'iou': torchmetrics.classification.BinaryJaccardIndex(),          
                'dice': torchmetrics.classification.BinaryF1Score()     
            })
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

    import torch
    from typing import Union, Dict
    from torch.utils.data import DataLoader
    import lightning.pytorch as pl


    @torch.no_grad()
    def evaluate(self, data: Union[DataLoader, CrowdDataModule]) -> Dict[str, float]:
        if isinstance(data, pl.LightningDataModule):
            data.setup(stage="test")
            loader = data.test_dataloader() or data.final_val_dataloader()
        else:
            loader = data
        
        # 1. Move metrics to the correct device and reset
        self.metrics = self.metrics.to(self.device)
        self.metrics.reset()
        
        if hasattr(self, 'mask_metrics') and self.mask_metrics is not None:
            self.mask_metrics = self.mask_metrics.to(self.device)
            self.mask_metrics.reset()
            
        # Ensure model is in eval mode
        self.model.eval()
        
        for batch in loader:
            # Unpack dynamic batch including original sizes
            # torch.cuda.empty_cache()
            x, y, orig_sizes = batch
            x, y = x.to(self.device), y.to(self.device)
            
            # 2. Forward Pass
            if getattr(self.model.params, 'loss_function', '') == 'mask_mse_ssim':
                pred_density, mask_logits = self.model(x, return_mask=True)
                mask_probs = torch.sigmoid(mask_logits)
                gt_mask = (y > 0).float()
            else:
                pred_density = self.model(x, return_mask=False)
                mask_probs = None
                gt_mask = None

            # 3. Unpad outputs for accurate evaluation
            pred_counts = []
            gt_counts = []
            
            unpadded_mask_probs = []
            unpadded_gt_masks = []
            
            for i in range(x.size(0)):
                h, w = orig_sizes[i][0], orig_sizes[i][1]
                
                real_pred = pred_density[i, ..., :h, :w]
                real_gt = y[i, ..., :h, :w]
                
                pred_counts.append(torch.sum(real_pred))
                gt_counts.append(torch.sum(real_gt))
                
                if mask_probs is not None:
                    real_mask_prob = mask_probs[i, ..., :h, :w]
                    real_gt_mask = gt_mask[i, ..., :h, :w]
                    
                    unpadded_mask_probs.append(real_mask_prob.flatten())
                    unpadded_gt_masks.append(real_gt_mask.flatten())
                    
            # 4. Stack counts
            pred_count_tensor = torch.stack(pred_counts)
            gt_count_tensor = torch.stack(gt_counts)
            
            # 5. Update Metrics
            self.metrics.update(pred_count_tensor, gt_count_tensor)
            
            if mask_probs is not None and hasattr(self, 'mask_metrics'):
                batch_mask_probs = torch.cat(unpadded_mask_probs)
                batch_gt_masks = torch.cat(unpadded_gt_masks)
                self.mask_metrics.update(batch_mask_probs, batch_gt_masks)

        # 6. Compute final results
        results = self.metrics.compute()
        final_metrics = {k: v.item() for k, v in results.items()}
        
        if mask_probs is not None and hasattr(self, 'mask_metrics'):
            mask_results = self.mask_metrics.compute()
            for k, v in mask_results.items():
                final_metrics[f"mask_{k}"] = v.item()
                
        return final_metrics

    # @torch.no_grad()
    # def evaluate(self, data: Union[DataLoader, pl.LightningDataModule]) -> Dict[str, float]:
    #     """
    #     Runs evaluation and returns a dictionary of all computed metrics (e.g., MAE, MSE).
    #     """
    #     if isinstance(data, pl.LightningDataModule):
    #         data.setup(stage="test")
    #         loader = data.test_dataloader() or data.val_dataloader()
    #     else:
    #         loader = data
        
    #     # Reset metrics to ensure a clean slate for this run
    #     self.metrics.reset()
    #     validator = pl.Trainer(enable_progress_bar=False,
    #             precision="bf16-mixed",
    #             accelerator='auto',
    #             )
    #     # val_results = trainer.validate(model, datamodule=self.datamodule, verbose=False)[0]
    #     results = validator.validate(BaseLitModel(self.model.params, self.model),
    #             dataloaders=loader, 
    #             verbose=False, 
    #             )[0]
    #     # for batch_x, batch_y in loader:
    #     #     batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
    #     #     # 1. Forward pass
    #     #     pred_density = self.model.sliding_window_inference(batch_x)
    #     #     # pred_density = F.relu(pred_density)
            
    #     #     # 2. Calculate the counts by summing across spatial and channel dimensions
    #     #     # Assuming shape is [Batch, Channel, Height, Width]
    #     #     pred_count = pred_density.sum(dim=(1, 2, 3))
    #     #     gt_count = batch_y.sum(dim=(1, 2, 3))
    #     #     print(pred_count, gt_count)
            
    #     #     # 3. MetricCollection updates all metrics simultaneously
    #     #     self.metrics.update(pred_count, gt_count)

    #     # compute() returns a dict: {'MAE': tensor(12.5), 'MSE': tensor(150.2)}
    #     # results = self.metrics.compute()
        
    #     # Convert tensors to standard python floats for the final return
    #     return results.items()