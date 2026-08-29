from typing import Optional

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
from torchvision.transforms import v2

from src import config
from src.core.params import BaseParams
from src.data.transform import UnpadToOriginal
from src.models.loss import MaskMSESSIMLoss, MSESSIMLoss, SSIMLoss, CountPenaltyLoss, SpatiallyWeightedLoss
from src.models.model import CrowdCounter
from src.models.metrics import MeanBiasError, PositiveOnlyNAE, CombinedMAEMBE
# from src.utils.registries import MODEL_REGISTRY


class BaseLitModel(pl.LightningModule):
    """Lightning wrapper that coordinates model forward passes, loss computation, and metrics.

    This module owns the crowd-counting architecture, selects the configured loss function,
    tracks training and validation metrics, and exposes inference hooks that integrate with
    the project's custom density-map pipeline.

    Attributes:
        params (BaseParams): Runtime configuration for the current training run.
        model (CrowdCounter): Crowd-counting network instance.
        criterion: Loss module selected from the configured loss-function enum.
        train_metrics: Metric collection for training counts.
        val_metrics: Metric collection for validation counts.
        train_mask_metrics: Metric collection for training foreground masks.
        val_mask_metrics: Metric collection for validation foreground masks.
    """
    def __init__(self, params: BaseParams, model: Optional[torch.nn.Module] = None):
        """Initialize the Lightning module and attach its training loss and metrics.

        Args:
            params (BaseParams): Parameter object containing the selected architecture and
                training configuration.
            model (Optional[torch.nn.Module]): Optional pre-instantiated model override.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.params = params
        if getattr(self, 'model', None) is None:
            self.model = CrowdCounter(self.params)
            # self.model = torch.compile(self.model, mode='default')

        # Shared Metrics
        metrics = torchmetrics.MetricCollection({
            'mae': torchmetrics.MeanAbsoluteError(),
            'rmse': torchmetrics.MeanSquaredError(squared=False),
            'nae': PositiveOnlyNAE(),
            'mbe': MeanBiasError(),
            'mae_mbe': CombinedMAEMBE(),
        })
        mask_metrics = torchmetrics.MetricCollection({
            'iou': torchmetrics.classification.BinaryJaccardIndex(),          
            'dice': torchmetrics.classification.BinaryF1Score()     
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.train_mask_metrics = mask_metrics.clone(prefix='train_mask_')
        self.val_mask_metrics = mask_metrics.clone(prefix='val_mask_')
        self.unpad = UnpadToOriginal()
        match self.params.loss_function:
            case 'mask_mse_ssim':
                self.criterion = MaskMSESSIMLoss(self.params)
            case 'count_penalty':
                self.criterion = CountPenaltyLoss(self.params)
            case 'spatially_weighted_loss':
                self.criterion = SpatiallyWeightedLoss(self.params)
            case'mse_ssim':
                self.criterion = MSESSIMLoss(self.params)
            case 'ssim':
                self.criterion = SSIMLoss()
            case _:
                self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        """Execute a forward pass through the underlying crowd-counting model.

        Args:
            x (torch.Tensor): Input batch tensor shaped ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Output density map or model-predicted value.
        """
        # x = self.transform(x)
        return self.model(x)
    
    def _shared_step(self, x, y):
        """Compute the model output, loss, and auxiliary mask tensors for train/validation.

        Args:
            x (torch.Tensor): Input image batch.
            y (torch.Tensor): Ground-truth density map batch.

        Returns:
            tuple: ``(loss, pred_density, mask_probs, gt_mask, loss_dict)`` where the mask
                tensors are optional and may be ``None`` when the chosen loss does not use
                a foreground mask.
        """
        mask_probs = None
        gt_mask = None
        
        # Forward Pass
        if self.params.loss_function == 'mask_mse_ssim':
            pred_density, mask_logits = self.model(x, return_mask=True)
            loss, loss_dict = self.criterion(pred_density, mask_logits, y)
            
            mask_probs = torch.sigmoid(mask_logits)
            gt_mask = (y > 0).float()
        else:
            pred_density = self.model(x, return_mask=False)
            loss = self.criterion(pred_density, y)
            loss_dict = None
        return loss, pred_density, mask_probs, gt_mask, loss_dict

    def training_step(self, batch, batch_idx):
        """Execute one training step and log count and mask metrics.

        Args:
            batch: A tuple of ``(x, y)`` where ``x`` is the input image batch and ``y``
                is the ground-truth density map batch.
            batch_idx (int): Index of the current minibatch.

        Returns:
            torch.Tensor: Scalar training loss used by Lightning.
        """
        # 1. Unpack standard training batch (Cropped uniformly, no sizes passed)
        x, y = batch 
        
        # 2. Run shared logic
        loss, pred_density, mask_probs, gt_mask, loss_dict = self._shared_step(x, y)
            
        # 3. Sum counts directly (No unpadding needed for training)
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / self.params.label_scaler
        gt_count = torch.sum(y, dim=(1, 2, 3)) / self.params.label_scaler
        
        # 4. Log Training Metrics
        self.train_metrics(pred_count, gt_count)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if mask_probs is not None and gt_mask is not None:
            self.train_mask_metrics(mask_probs, gt_mask)
            self.log_dict(self.train_mask_metrics, on_step=False, on_epoch=True, prog_bar=True)
            if loss_dict:
                train_loss_logs = {f"train_{k}": v for k, v in loss_dict.items()}
                self.log_dict(train_loss_logs, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Execute one validation step and compute metrics on the original image scale.

        Args:
            batch: A tuple of ``(x, y, orig_sizes)`` containing padded image batches,
                targets, and the original spatial sizes for each item.
            batch_idx (int): Index of the current validation minibatch.

        Returns:
            None: Validation metrics are logged to the trainer via ``self.log``.
        """
        # 1. Unpack dynamic batch including original sizes from collate_fn
        x, y, orig_sizes = batch
        
        # 2. Run shared logic (Loss is computed safely on padded tensors)
        loss, pred_density, mask_probs, gt_mask, loss_dict = self._shared_step(x, y)

        # 3. Unpad outputs for accurate evaluation
        pred_counts = []
        gt_counts = []
        
        unpadded_mask_probs = []
        unpadded_gt_masks = []
        
        for i in range(x.size(0)):
            # Crop density maps back to real image dimensions
            real_pred = self.unpad(pred_density[i], orig_sizes[i])
            real_gt = self.unpad(y[i], orig_sizes[i])
            
            pred_counts.append(torch.sum(real_pred))
            gt_counts.append(torch.sum(real_gt))
            
            # Crop masks back to real dimensions to avoid calculating IoU on padded space
            if mask_probs is not None:
                real_mask_prob = self.unpad(mask_probs[i], orig_sizes[i])
                real_gt_mask = self.unpad(gt_mask[i], orig_sizes[i])
                
                unpadded_mask_probs.append(real_mask_prob.flatten())
                unpadded_gt_masks.append(real_gt_mask.flatten())
                
        # 4. Stack counts and scale
        pred_count_tensor = torch.stack(pred_counts)
        gt_count_tensor = torch.stack(gt_counts)
        
        # 5. Log Validation Count Metrics & Loss
        self.val_metrics(pred_count_tensor, gt_count_tensor)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('epoch_idx', float(self.current_epoch), on_step=False, on_epoch=True)
        
        # 6. Log Validation Mask Metrics
        if mask_probs is not None:
            batch_mask_probs = torch.cat(unpadded_mask_probs)
            batch_gt_masks = torch.cat(unpadded_gt_masks)
            
            self.val_mask_metrics(batch_mask_probs, batch_gt_masks)
            self.log_dict(self.val_mask_metrics, on_step=False, on_epoch=True, prog_bar=True)
            if loss_dict:
                val_loss_logs = {f"val_{k}": v for k, v in loss_dict.items()}
                self.log_dict(val_loss_logs, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        """Create the optimizer and optional scheduler for the current training run.

        Returns:
            torch.optim.Optimizer | dict: Optimizer instance or PyTorch Lightning
                scheduler dictionary.
        """
        # 1. Group parameters
        param_groups = self._get_param_groups()
        
        # 2. Build optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.params.l2_reg
        )

        # 3. Build scheduler (returns None if no scheduler is needed)
        scheduler_config = self._get_scheduler_config(optimizer)

        # 4. Return standard PyTorch Lightning format
        if scheduler_config:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_config
            }
            
        return optimizer
    
    
    
    def predict(self, x):
        """Run inference for a single image or batch using the wrapped crowd-counter.

        Args:
            x (torch.Tensor): Input images expected to match the model's normalized
                ``(B, C, H, W)`` format.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Total count estimates and density maps.
        """
        import src.core.inference as inference
        return inference.predict(self.model, x, self.device)
    

    
    def _get_param_groups(self):
        """Partition trainable parameters into backbone, head, and auxiliary loss groups.

        Returns:
            list[dict]: A list of optimizer parameter groups with individually tuned
                learning rates.
        """
        backbone_params = []
        head_params = []
        loss_params = []
        
        # FIX: Change self.model.named_parameters() to self.named_parameters()
        # This ensures it searches the ENTIRE LightningModule, including self.criterion
        for name, param in self.named_parameters():
            if 'criterion' in name or 'log_vars' in name:
                # Catch the learnable loss weights
                loss_params.append(param)
            elif 'backbone' in name or 'encoder' in name:
                # Catch the backbone/encoder
                backbone_params.append(param)
            else:
                # Catch everything else (the prediction head, etc.)
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': self.params.lr * 0.1}, 
            {'params': head_params, 'lr': self.params.lr}            
        ]
        
        # Add the loss parameters if they exist
        if loss_params:
            param_groups.append({
                'params': loss_params, 
                'lr': self.params.lr, 
                'weight_decay': 0.0  # Best practice: don't apply weight decay to loss variances
            })

        return param_groups
    
    def _get_scheduler_config(self, optimizer):
        """Build the scheduler configuration based on the selected learning-rate strategy.

        Args:
            optimizer: PyTorch optimizer instance.

        Returns:
            Optional[dict]: Scheduler configuration compatible with Lightning, or ``None``
                when no scheduler is selected.
        """
        match self.params.lr_schedule:
            case 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    **self.params.scheduler_kwargs
                )
                return {
                    "scheduler": scheduler,
                    "monitor": self.params.monitor_metric, 
                    "interval": "epoch"   
                }

            case 'clipped_exp':
                decay_rate = self.params.scheduler_kwargs.get('decay_rate', 0.96)
                min_lr = self.params.lr * self.params.scheduler_kwargs.get('min_lr_pct', 0.01)
                decay_steps = self.params.scheduler_kwargs.get('decay_steps', max(1, self.params.train_size // self.params.batch_size))

                lr_lambda = lambda step: max(
                    (decay_rate ** (step / decay_steps)), 
                    min_lr / self.params.lr 
                )
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return {
                    "scheduler": scheduler, 
                    "interval": "step"
                }
            case 'multi_step':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    **self.params.scheduler_kwargs
                )
                return {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            
        return None