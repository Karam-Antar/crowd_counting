from typing import Optional
import torch
import lightning.pytorch as pl

from src.models.lit_model import BaseLitModel
from src.core.params import BaseParams
from src.utils.experiment_trackers import BaseTracker
from src.utils.model_registry import BaseRegistry, MLFlowRegistry, ModelPayload
from src.experiment.base import BaseExperimentRunner


class StandardRunner(BaseExperimentRunner):
    """Subclass for standard single-run execution."""

    def __init__(
        self,
        model_cls: type[torch.nn.Module],
        tracker: BaseTracker,
        monitor_metric: str = "val_nae", 
        monitor_mode: str = "min",
        lit_model_cls: type[pl.LightningModule] = BaseLitModel,
        registry: Optional[BaseRegistry] = MLFlowRegistry(),
        params: Optional[BaseParams] = None,
        ckpt_path: Optional[str] = None
    ):
        super().__init__(model_cls, monitor_metric, monitor_mode, lit_model_cls)
        self.tracker = tracker
        self.registry = registry
        self.params = params
        self.ckpt_path = ckpt_path


    def _build_model(self) -> pl.LightningModule:
        if self.ckpt_path:
            return self.lit_model_cls.load_from_checkpoint(self.ckpt_path, weights_only=False)
        return self.lit_model_cls(self.params)

    def run(self):
        """Executes a single training run using a pre-configured Tracker instance."""
        self.tracker.start_run()

        try:
            payload = self._run_training()
            lit_model, best_path, metrics = payload.model, payload.ckpt_path, payload.metrics
            if self.registry:
                self.registry.upload_model(payload)
        except Exception as e:
            print(e)
            self.tracker.end_run('failed')
            raise e
        
        self.tracker.end_run('success')
            
        val_metrics = {k: v for k,v in metrics.items() if 'val' in k}
        train_metrics = {k: v for k,v in metrics.items() if 'train' in k}
        
        return lit_model, best_path, {"val": val_metrics}, {"train": train_metrics}
