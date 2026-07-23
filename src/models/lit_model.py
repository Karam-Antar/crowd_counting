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
# from src.utils.registries import MODEL_REGISTRY


class BaseLitModel(pl.LightningModule):
    """Lightning module that uses composition to wrap a torch Module."""
    def __init__(self, params: BaseParams, model: Optional[torch.nn.Module] = None):
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
            'nae': torchmetrics.MeanAbsolutePercentageError()
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
        # x = self.transform(x)
        return self.model(x)
    
    def _shared_step(self, x, y):
        """Handles the forward pass, loss computation, and mask preparation for both train and val."""
        mask_probs = None
        gt_mask = None
        
        # Forward Pass
        if self.params.loss_function == 'mask_mse_ssim':
            pred_density, raw_density, mask_logits = self.model(x, return_mask=True)
            loss = self.criterion(raw_density, mask_logits, y)
            
            mask_probs = torch.sigmoid(mask_logits)
            gt_mask = (y > 0).float()
        else:
            pred_density = self.model(x, return_mask=False)
            loss = self.criterion(pred_density, y)
            
        return loss, pred_density, mask_probs, gt_mask

    def training_step(self, batch, batch_idx):
        # 1. Unpack standard training batch (Cropped uniformly, no sizes passed)
        x, y = batch 
        
        # 2. Run shared logic
        loss, pred_density, mask_probs, gt_mask = self._shared_step(x, y)
            
        # 3. Sum counts directly (No unpadding needed for training)
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / self.params.label_scaler
        gt_count = torch.sum(y, dim=(1, 2, 3)) / (self.params.label_scaler + 1e-6)
        
        # 4. Log Training Metrics
        self.train_metrics(pred_count, gt_count)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if mask_probs is not None and gt_mask is not None:
            self.train_mask_metrics(mask_probs, gt_mask)
            self.log_dict(self.train_mask_metrics, on_step=False, on_epoch=True, prog_bar=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        # 1. Unpack dynamic batch including original sizes from collate_fn
        x, y, orig_sizes = batch
        
        # 2. Run shared logic (Loss is computed safely on padded tensors)
        loss, pred_density, mask_probs, gt_mask = self._shared_step(x, y)

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

    def configure_optimizers(self):
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
        """
        Custom inference method for single inputs.
        """
        import src.core.inference as inference
        return inference.predict(self.model, x, self.device)
    

    
    def _get_param_groups(self):
        """Splits model parameters into backbone and head groups with different LRs."""
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'encoder' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': backbone_params, 'lr': self.params.lr * 0.1}, 
            {'params': head_params, 'lr': self.params.lr}            
        ]
    
    def _get_scheduler_config(self, optimizer):
        """Returns the Lightning scheduler dictionary based on config, or None."""
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