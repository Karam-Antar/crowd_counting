import os
import datetime
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import LitLogger
from optuna.integration import PyTorchLightningPruningCallback
from typing import Optional, Any, Callable
import torch
import litlogger
from src.data.datamodule import CrowdDataModule
from src.models.lit_model import BaseLitModel
from src.core.params import BaseParams
from src.core.callbacks import ReseedCallback
from src.utils import helpers
from src.utils.model_registry import upload_model_to_lit
from src.utils.helpers import get_study_best_value
# from src.utils.model_persistence import save_model
from src import config


class Experiment:
    """PyTorch Lightning trainer wrapper for flower image classification."""

    def __init__(
        self,
        experiment_name: str,
        architecture: str,
        model_cls: type[torch.nn.Module],
        monitor_metric: str = "val_nae", 
        monitor_mode: str = "min",
        lit_model_cls: type[pl.LightningModule] = BaseLitModel,
    ):
        self.experiment_name = experiment_name
        self.architecture = architecture
        self.model_cls = model_cls
        self.lit_model_cls = lit_model_cls
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

    def _build_model(self, params: BaseParams, ckpt_path: Optional[str] = None) -> pl.LightningModule:
        if ckpt_path:
            return self.lit_model_cls.load_from_checkpoint(ckpt_path, weights_only=False)
        return self.lit_model_cls(params, self.model_cls(params))

    def _evaluate_model(self, trainer: pl.Trainer, model: pl.LightningModule) -> dict:
        # Get the dictionary of metrics from the first (and usually only) dataloader
        val_results = trainer.validate(model, datamodule=self.datamodule, verbose=False)[0]
        train_results = trainer.validate(model, dataloaders=self.datamodule.train_eval_dataloader(), verbose=False)[0]

        final_metrics = {}

        # 1. Process Validation Metrics
        for metric_name, value in val_results.items():
            final_metrics[f"best_{metric_name}"] = value

        # 2. Process Training Metrics
        for metric_name, value in train_results.items():
            # Because we used trainer.validate(), the keys will still start with 'val_'.
            # We need to replace 'val_' with 'train_' to accurately reflect the data source.
            if metric_name.startswith("val_"):
                train_key = metric_name.replace("val_", "train_", 1)
            else:
                train_key = f"train_{metric_name}"
                
            final_metrics[f"best_{train_key}"] = value

        return final_metrics
    
    def _create_logger(self, name: str):
        # logger = LitLogger(name=name, save_logs=False)
        # metadata = logger.experiment.metadata
        lit_experiment = litlogger.init(name=name)
        metadata = lit_experiment.metadata
        # # print('metadata', metadata)
        # # print(len(metadata))
        if len(metadata):
            lit_experiment.finalize('aborted')
            # if hasattr(logger.experiment, 'finalize'):
            #     logger.experiment.finalize('aborted')
            logger = LitLogger(name=helpers.get_unique_experiment_name(name), save_logs=False)
        else:
            logger = LitLogger(name=name, save_logs=False)
        return logger

    def _upload_model_artifact(self, lit_model: pl.LightningModule, model_name: str, lit_experiment: litlogger.Experiment, params_dict: dict, ckpt_path: str | None = None):
        """Helper to upload the model to the Lit Platform."""
        upload_model_to_lit(
            lit_model,
            model_name, 
            lit_experiment, 
            params_dict, 
            code_artifacts=None,
            ckpt_path=ckpt_path
        )
    
    def _get_default_callbacks(self) -> list:
        """Generates the standard callbacks dynamically based on the injected metric."""
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

    def _run_training(
        self, 
        params: BaseParams, 
        run_name: str,
        lit_experiment_path: str, 
        ckpt_path: Optional[str] = None,
        optuna_trial: Optional[optuna.Trial] = None
    ):
        lit_model = self._build_model(params, ckpt_path)
        self.datamodule = CrowdDataModule(params=params)
        
        pl_logger = self._create_logger(f'{lit_experiment_path}/{run_name}')
        pl_logger.log_hyperparams(params.to_dict(flatten=True, to_str=True))

        callbacks = self._get_default_callbacks()
        
        if optuna_trial:
            callbacks.append(PyTorchLightningPruningCallback(optuna_trial, monitor=self.monitor_metric))

        trainer = pl.Trainer(
            max_epochs=params.epochs,
            logger=pl_logger,
            callbacks=callbacks,
            enable_progress_bar=False,
            accelerator='auto',
            log_every_n_steps=1,
            limit_train_batches=1,
            limit_val_batches=1,
        )

        trainer.fit(lit_model, datamodule=self.datamodule)

        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            lit_model = self._build_model(params, ckpt_path=best_path)

        metrics = self._evaluate_model(trainer, lit_model)

        lit_experiment = litlogger.init(name=pl_logger.name)
        lit_experiment.log_metrics(metrics)
        lit_experiment.log_metadata(params.to_dict(flatten=True, to_str=True))

        return lit_model, best_path, metrics, lit_experiment, params

    def fit(self, params: BaseParams, run_name: Optional[str] = None, ckpt_path: Optional[str] = None):
        """Standard single-run execution."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.experiment_name}_{self.architecture}"
        safe_run_name = run_name or f"run_{timestamp}"
        lit_experiment_path = f'{self.experiment_name}/{self.architecture}'
        
        lit_model, path, metrics, lit_exp, params = self._run_training(params, safe_run_name, lit_experiment_path, ckpt_path)
        
        # For a single fit, we always upload
        if path:
            self._upload_model_artifact(lit_model.model, model_name,  lit_exp, params, ckpt_path=path)
        lit_exp.finalize('success')
            
        target_val = metrics.get(f"best_{self.monitor_metric}", None)
        target_train = metrics.get(f"best_train_{self.monitor_metric.replace('val_', '')}", None)
        val_metrics = {k: v for k,v in metrics.items() if 'val' in k}
        train_metrics = {k: v for k,v in metrics.items() if 'train' in k}
        
        return lit_model, path, {"val": val_metrics}, {"train": train_metrics}

    def optimize(self, study_name_suffix: str, params_cls: type[BaseParams] = BaseParams, n_trials: int = 12) -> optuna.Study:

        lit_experiment_path = f"{self.experiment_name}_studies/{self.architecture}/{study_name_suffix}"
        
        def objective(trial: optuna.Trial) -> float:
            run_name = f"trial_{trial.number}"
            
            lit_model, best_path, metrics, lit_exp, params = self._run_training(
                params=params_cls.suggest(trial), # <-- Changed from self.params_cls
                run_name=run_name,
                lit_experiment_path=lit_experiment_path, 
                optuna_trial=trial
            )
            
            # --- NEW: Use dynamic metric for optimization ---
            target_metric_value = metrics.get(f"best_{self.monitor_metric}")
            
            trial.set_user_attr(f"best_{self.monitor_metric}", target_metric_value)
            
            # Optuna Logic for uploading best models
            current_best = get_study_best_value(trial)
            
            # Handle min vs max logic dynamically
            is_better = (target_metric_value > current_best) if self.monitor_mode == "max" else (target_metric_value < current_best)
            
            if is_better:
                trial.study.set_user_attr(f"best_{self.monitor_metric}", target_metric_value)
                if best_path:
                    artifact_name = f"{self.experiment_name}_{self.architecture}_{study_name_suffix}"
                    self._upload_model_artifact(lit_model.model, artifact_name, lit_exp, params, ckpt_path=best_path)
            
            lit_exp.finalize('success')
            return target_metric_value

        # --- NEW: Set direction based on user input ---
        study_direction = "maximize" if self.monitor_mode == "max" else "minimize"

        db_url = os.environ.get('OPTUNA_DB_URL', 'sqlite:///optuna.db')
        study = optuna.create_study(
            study_name=lit_experiment_path,
            storage=db_url,
            load_if_exists=True,
            direction=study_direction,
            sampler=optuna.samplers.TPESampler(seed=config.SEED),
        )

        study.optimize(objective, n_trials=n_trials)
        return study