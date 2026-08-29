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

class ModelWrapper:
    """Wrap a compiled or exported crowd-counting model for evaluation and inference.

    The wrapper keeps the device-aware model instance, configures evaluation metrics,
    and provides a consistent inference interface for both TorchScript/exported models
    and native PyTorch modules. It is intended for production-style validation and
    batch metric aggregation for crowd counting tasks.

    Attributes:
        exported_program (Optional[ExportedProgram]): Original exported model artifact,
            if the wrapper was initialized from a Torch Export program.
        device (torch.device): Device used for inference and metric updates.
        model (CrowdCounter): The concrete model loaded into the selected device.
        metrics (MetricCollection): Count-based evaluation metrics.
        mask_metrics (Optional[MetricCollection]): Optional segmentation metrics for
            foreground mask quality when the model emits a mask head.
    """
    def __init__(self, exported_program: ExportedProgram | GraphModule, device: str = config.device, metrics: Optional[MetricCollection] = None):
        """Initialize a model wrapper around an exported PyTorch program or module.

        Args:
            exported_program (ExportedProgram | GraphModule): A compiled model artifact
                or a graph module instance to wrap for inference and evaluation.
            device (str): Target device string such as ``"cuda"`` or ``"cpu"``.
            metrics (Optional[MetricCollection]): Optional metric collection to use
                instead of the default crowd-counting metrics.

        Raises:
            TypeError: If the provided model object is neither an exported program nor
                a valid graph module.
        """
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
        """Run inference on a single image and return density count plus map.

        Args:
            x (torch.Tensor): Input image or batch with shape ``(C, H, W)`` or
                ``(B, C, H, W)``.

        Returns:
            tuple[float, torch.Tensor]: The total count estimate and the corresponding
            density map tensor.

        Raises:
            ValueError: If the input tensor cannot be processed by the underlying model.
        """
        return inference.predict(self.model, x)
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        """Forward a tensor through the wrapped model without extra wrapper logic.

        Args:
            x (torch.Tensor): Input batch tensor expected to match the model's
                training image layout.

        Returns:
            torch.Tensor: The model's raw output tensor.
        """
        return self.model(x)


    @torch.no_grad()
    def evaluate(self, data: Union[DataLoader, "CrowdDataModule"]) -> Dict[str, float]:
        """Evaluate the model across a dataloader or Lightning datamodule.

        Args:
            data (Union[DataLoader, CrowdDataModule]): Batched evaluation data. When a
                LightningDataModule is supplied, it is initialized for the test stage.

        Returns:
            Dict[str, float]: Aggregated evaluation metrics converted to Python floats.

        Raises:
            RuntimeError: If the dataloader output does not contain the expected image,
                target, and original-size tuple structure.
        """
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
        
        # Safely extract the device type string ('cuda' or 'cpu') for autocast
        device_type = self.device.type if isinstance(self.device, torch.device) else torch.device(self.device).type

        for batch in loader:
            # Unpack dynamic batch including original sizes
            x, y, orig_sizes = batch
            x, y = x.to(self.device), y.to(self.device)
            
            # 2. Forward Pass wrapped in Mixed Precision (bf16)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if getattr(self.model.params, 'loss_function', '') == 'mask_mse_ssim':
                    pred_density, mask_logits = self.model(x, return_mask=True)
                    mask_probs = torch.sigmoid(mask_logits)
                    gt_mask = (y > 0).float()
                else:
                    pred_density = self.model(x, return_mask=False)
                    mask_probs = None
                    gt_mask = None

            # 3. Unpad outputs for accurate evaluation 
            # (These operations and metric calculations will happen in standard precision)
            pred_counts = []
            gt_counts = []
            
            unpadded_mask_probs = []
            unpadded_gt_masks = []
            
            for i in range(x.size(0)):
                h, w = orig_sizes[i][0], orig_sizes[i][1]
                
                real_pred = pred_density[i, ..., :h, :w]
                real_gt = y[i, ..., :h, :w]
                
                # Using float() here ensures we accumulate counts in fp32 to avoid bf16 precision limits on large sums
                pred_counts.append(torch.sum(real_pred).float())
                gt_counts.append(torch.sum(real_gt).float())
                
                if mask_probs is not None:
                    real_mask_prob = mask_probs[i, ..., :h, :w]
                    real_gt_mask = gt_mask[i, ..., :h, :w]
                    
                    unpadded_mask_probs.append(real_mask_prob.flatten().float())
                    unpadded_gt_masks.append(real_gt_mask.flatten().float())
                    
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