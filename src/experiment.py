import os
import datetime
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from typing import Optional, Type
import torch

from src.data.datamodule import CrowdDataModule
from src.models.lit_model import BaseLitModel
from src.core.params import BaseParams
from src.core.callbacks import ReseedCallback
from src.utils.experiment_trackers import BaseTracker, MLFlowTracker
from src.utils.helpers import get_study_best_value
from src import config
from abc import ABC, abstractmethod


class BaseExperiment(ABC):
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

    def _build_model(self) -> pl.LightningModule:
        return self.lit_model_cls(self.params)

    def _evaluate_model(self, trainer: pl.Trainer, model: pl.LightningModule) -> dict:
        val_results = trainer.validate(model, datamodule=self.datamodule, verbose=False)[0]
        train_results = trainer.validate(model, dataloaders=self.datamodule.train_eval_dataloader(), verbose=False)[0]

        final_metrics = {}
        for metric_name, value in val_results.items():
            final_metrics[f"best_{metric_name}"] = value

        for metric_name, value in train_results.items():
            train_key = metric_name.replace("val_", "train_", 1) if metric_name.startswith("val_") else f"train_{metric_name}"
            final_metrics[f"best_{train_key}"] = value

        return final_metrics

    def _get_default_callbacks(self) -> list:
        return [
            EarlyStopping(monitor=self.monitor_metric, patience=50, mode=self.monitor_mode),
            ReseedCallback(),
            ModelCheckpoint(
                monitor=self.monitor_metric, 
                mode=self.monitor_mode, 
                save_top_k=1, 
                dirpath="./checkpoints"
            )
        ]

    def _run_training(self):
        lit_model = self._build_model()
        self.datamodule = CrowdDataModule(params=self.params)
        
        pl_logger = self.tracker.get_logger()
        pl_logger.log_hyperparams(self.params.to_dict(flatten=True, to_str=True))

        callbacks = self._get_default_callbacks()

        trainer = pl.Trainer(
            max_epochs=self.params.epochs,
            logger=pl_logger,
            callbacks=callbacks,
            enable_progress_bar=False,
            accelerator='auto',
            log_every_n_steps=1,
            limit_train_batches=2,
            limit_val_batches=2,
        )

        trainer.fit(lit_model, datamodule=self.datamodule)

        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            lit_model = self.lit_model_cls.load_from_checkpoint(best_path, weights_only=False)

        metrics = self._evaluate_model(trainer, lit_model)

        self.tracker.log_results(metrics, self.params.to_dict(flatten=True, to_str=True))

        return lit_model, best_path, metrics
    
    @abstractmethod
    def run(self):
        pass


class StandardRunner(BaseExperiment):
    """Subclass for standard single-run execution."""

    def __init__(
        self,
        model_cls: type[torch.nn.Module],
        tracker: BaseTracker,
        monitor_metric: str = "val_nae", 
        monitor_mode: str = "min",
        lit_model_cls: type[pl.LightningModule] = BaseLitModel,
        params: Optional[BaseParams] = None,
        ckpt_path: Optional[str] = None
    ):
        super().__init__(model_cls, monitor_metric, monitor_mode, lit_model_cls)
        self.tracker = tracker
        self.params = params
        self.ckpt_path = ckpt_path


    def _build_model(self) -> pl.LightningModule:
        if self.ckpt_path:
            return self.lit_model_cls.load_from_checkpoint(self.ckpt_path, weights_only=False)
        return self.lit_model_cls(self.params)

    def run(self):
        """Executes a single training run using a pre-configured Tracker instance."""
        self.tracker.start_run()

        lit_model, path, metrics = self._run_training()
        
        self.tracker.end_run('success')
            
        val_metrics = {k: v for k,v in metrics.items() if 'val' in k}
        train_metrics = {k: v for k,v in metrics.items() if 'train' in k}
        
        return lit_model, path, {"val": val_metrics}, {"train": train_metrics}


class OptunaTuner(BaseExperiment):
    """Subclass for Optuna hyperparameter optimization."""


    def __init__(
        self,
        experiment: str,
        model_cls: type[torch.nn.Module],
        monitor_metric: str = "val_nae", 
        monitor_mode: str = "min",
        lit_model_cls: type[pl.LightningModule] = BaseLitModel,
    ):
        super().__init__(model_cls, monitor_metric, monitor_mode, lit_model_cls)
        self.experiment = experiment


    def _get_default_callbacks(self):
        callbacks =  super()._get_default_callbacks()
        callbacks.append(PyTorchLightningPruningCallback(self.current_trial, monitor=self.monitor_metric))
        return callbacks
    
    @property
    def _best_value(self):
        """Get the best validation accuracy from the study."""
        best_value = -float("inf")
        try:
            best_value = self.current_trial.study.user_attrs.get("val_accuracy", -float("inf"))
        except:
            pass
        return best_value


    def run(
        self, 
        study_name_suffix: str, 
        tracker_cls: type[BaseTracker], # Accept the blueprint class here
        params_cls: type[BaseParams] = BaseParams, 
        n_trials: int = 12
    ) -> optuna.Study:
        
        study_name = f"{self.experiment}/{study_name_suffix}"
        study_exp_name = f"{self.experiment}_studies"
        
        def objective(trial: optuna.Trial) -> float:
            run_name = f"trial_{trial.number}"
            
            # Spawn a new tracker instance for this trial
            self.tracker = tracker_cls(
                experiment=study_exp_name, 
                run_name=[study_name_suffix, run_name]
            )
            self.tracker.start_run()
            self.params = params_cls.suggest(trial)
            self.current_trial = trial
            
            lit_model, best_path, metrics = self._run_training()
            
            target_metric_value = metrics.get(f"best_{self.monitor_metric}")
            trial.set_user_attr(f"best_{self.monitor_metric}", target_metric_value)
            
            is_better = (target_metric_value > self._best_value) if self.monitor_mode == "max" else (target_metric_value < self._best_value)
            
            if is_better:
                trial.study.set_user_attr(f"best_{self.monitor_metric}", target_metric_value)
            
            self.tracker.end_run('success')
            return target_metric_value

        study_direction = "maximize" if self.monitor_mode == "max" else "minimize"
        db_url = os.environ.get('OPTUNA_DB_URL', 'sqlite:///optuna.db')
        study = optuna.create_study(
            study_name=study_name,
            storage=db_url,
            load_if_exists=True,
            direction=study_direction,
            sampler=optuna.samplers.TPESampler(seed=config.SEED),
        )

        study.optimize(objective, n_trials=n_trials)
            
        return study