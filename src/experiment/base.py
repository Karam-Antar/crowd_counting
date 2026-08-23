import os
import datetime
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from typing import Optional, Type
import torch
from abc import ABC, abstractmethod
from lightning.pytorch.callbacks import LearningRateMonitor
from src.data.datamodule import CrowdDataModule
from src.models.lit_model import BaseLitModel
from src.core.params import BaseParams
from src.utils.experiment_trackers import BaseTracker
from src.utils.model_registry import BaseRegistry
from src.utils.model_registry.utils import ModelPayload


class BaseExperimentRunner(ABC):
    """Base PyTorch Lightning engine handling model building and the core training loop."""

    def __init__(
        self,
        model_cls: type[torch.nn.Module],
        monitor_metric: str = "val_nae", 
        monitor_mode: str = "min",
        lit_model_cls: type[pl.LightningModule] = BaseLitModel,
    ):
        self.model_cls = model_cls
        self.lit_model_cls = lit_model_cls
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.params: Optional[BaseParams] = None
        self.tracker: Optional[BaseTracker] = None
        self.registry: Optional[BaseRegistry] = None

    def _build_model(self) -> pl.LightningModule:
        return self.lit_model_cls(self.params)

    def _evaluate_model(self, trainer: pl.Trainer, model: pl.LightningModule) -> dict:
        original_loggers = trainer.loggers
        trainer.loggers = []
        val_results = trainer.validate(model, datamodule=self.datamodule, verbose=False)[0]
        train_results = trainer.validate(model, dataloaders=self.datamodule.train_eval_dataloader(), verbose=False)[0]
        trainer.loggers = original_loggers

        final_metrics = {}
        for metric_name, value in val_results.items():
            final_metrics[f"best_{metric_name}"] = value

        for metric_name, value in train_results.items():
            train_key = metric_name.replace("val_", "train_", 1) if metric_name.startswith("val_") else f"train_{metric_name}"
            final_metrics[f"best_{train_key}"] = value

        return final_metrics

    def _get_default_callbacks(self) -> list:
        return [
            EarlyStopping(monitor=self.monitor_metric, patience=self.params.stop_patience, mode=self.monitor_mode),
            # ReseedCallback(),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                monitor=self.monitor_metric, 
                mode=self.monitor_mode, 
                save_top_k=1, 
                dirpath="./checkpoints"
            )
        ]

    def _run_training(self):
        self.datamodule = CrowdDataModule(params=self.params)
        self.datamodule.setup('fit')
        lit_model = self._build_model()
        logger = self.tracker.get_logger()
        self.tracker.log_init(self.params)

        callbacks = self._get_default_callbacks()

        trainer = pl.Trainer(
            max_epochs=self.params.epochs,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=False,
            accelerator='auto',
            accumulate_grad_batches=self.params.grad_accumulation,
            precision="bf16-mixed",
            gradient_clip_val=self.params.grad_clip,
            check_val_every_n_epoch=self.params.check_val_every_n_epoch,
            # log_every_n_steps=1,
            limit_train_batches=1,
            limit_val_batches=1,
        )

        trainer.fit(lit_model, datamodule=self.datamodule)

        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            lit_model = self.lit_model_cls.load_from_checkpoint(best_path, weights_only=False)

        metrics = self._evaluate_model(trainer, lit_model)

        self.tracker.log_results(metrics, self.params.to_dict(flatten=True, to_str=True))
        payload = ModelPayload(
            model=lit_model,
            tracker=self.tracker,
            params=self.params,
            metrics=metrics,
            monitor_metric=self.monitor_metric,
            monitor_mode=self.monitor_mode,
            # ckpt_path=best_path
        )
        return payload

    @abstractmethod
    def run(self):
        pass
