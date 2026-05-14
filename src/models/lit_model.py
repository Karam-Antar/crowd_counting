from typing import Optional

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
from torchvision.transforms import v2

from src import config
from src.core.params import BaseParams
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

        # Shared Metrics
        metrics = torchmetrics.MetricCollection({
            'mae': torchmetrics.MeanAbsoluteError(),
            'rmse': torchmetrics.MeanSquaredError(squared=False),
            'nae': torchmetrics.MeanAbsolutePercentageError()
        })
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
    
    def forward(self, x):
        # x = self.transform(x)
        return self.model(x)
    
    def _shared_step(self, batch):
        x, y = batch # x: Image, y: Density Map
        print(x.shape)
        
        # 2. Forward Pass
        preds = self.model(x)
        
        # 3. Loss: Mean Squared Error is standard for Density Maps
        loss = self.criterion(preds, y)
        
        # 4. Count-based Metrics
        # We compare the SUM of the maps (the actual person count)
        pred_count = torch.sum(preds, dim=(1, 2, 3))
        gt_count = torch.sum(y, dim=(1, 2, 3))
        
        return loss, pred_count, gt_count

    def training_step(self, batch, batch_idx):
        loss, pred_count, gt_count = self._shared_step(batch)
        
        # Update and log all metrics at once
        output = self.train_metrics(pred_count, gt_count)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_count, gt_count = self._shared_step(batch)
        
        # Update and log all metrics at once
        self.val_metrics(pred_count, gt_count)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    

    def configure_optimizers(self):
        # 1. Group parameters
        param_groups = self._get_param_groups()
        
        # 2. Build optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.params.l2_reg or 1e-4
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
        import core.inference as inference
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
        if self.params.lr_schedule == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                min_lr=self.params.lr * (self.params.min_lr_pct or 0.01)
            )
            return {
                "scheduler": scheduler,
                "monitor": "val_nae", 
                "interval": "epoch"   
            }

        if self.params.lr_schedule == 'clipped_exp':
            decay_rate = self.params.decay_rate or 0.96
            min_lr = self.params.lr * (self.params.min_lr_pct or 0.01)
            decay_steps = self.params.decay_steps or max(1, self.params.train_size // self.params.batch_size)

            lr_lambda = lambda step: max(
                (decay_rate ** (step / decay_steps)), 
                min_lr / self.params.lr 
            )
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "scheduler": scheduler, 
                "interval": "step"
            }
            
        return None