import os
from typing import Optional, Type
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import lightning.pytorch as pl

from src.models.lit_model import BaseLitModel
from src.core.params import BaseParams
from src.utils.experiment_trackers import BaseTracker, MLFlowTracker
from src.utils.model_registry import BaseRegistry, ModelPayload
from src import config
from src.experiment.base import BaseExperimentRunner


class OptunaTuner(BaseExperimentRunner):
    """Subclass for Optuna hyperparameter optimization."""

    def __init__(
        self,
        experiment: str,
        model_cls: type[torch.nn.Module],
        study_name_suffix: str, 
        monitor_metric: str = "val_nae", 
        monitor_mode: str = "min",
        lit_model_cls: type[pl.LightningModule] = BaseLitModel,
        tracker_cls: type[BaseTracker] = MLFlowTracker, 
        params_cls: type[BaseParams] = BaseParams, 
        n_trials: int = 12,
        registry_cls: Optional[type[BaseRegistry]] = None,
    ):
        super().__init__(model_cls, monitor_metric, monitor_mode, lit_model_cls)
        self.experiment = experiment
        self.study_name_suffix = study_name_suffix
        self.tracker_cls = tracker_cls
        self.params_cls = params_cls
        self.n_trials = n_trials
        self.registry_cls = registry_cls

    def _get_default_callbacks(self):
        callbacks = super()._get_default_callbacks()
        callbacks.append(PyTorchLightningPruningCallback(self.current_trial, monitor=self.monitor_metric))
        return callbacks
    
    @property
    def _best_value(self):
        """Get the best monitored metric value from the study so far, safely handling min/max modes."""
        default_val = -float("inf") if self.monitor_mode == "max" else float("inf")
        try:
            return self.current_trial.study.user_attrs.get(f"best_{self.monitor_metric}", default_val)
        except:
            return default_val

    def run(self) -> optuna.Study:
        study_name = f"{self.experiment}/{self.study_name_suffix}"
        study_exp_name = f"{self.experiment}_studies"
        
        # 1. Initialize a placeholder to capture the absolute best payload across all trials
        best_payload = None
        
        def objective(trial: optuna.Trial) -> float:
            # Gain write-access to the outer placeholder variable
            nonlocal best_payload
            run_name = f"trial_{trial.number}"
            
            # Spawn a new tracker instance for this trial
            self.tracker = self.tracker_cls(
                experiment=study_exp_name, 
                run_name=[self.study_name_suffix, run_name]
            )
            self.tracker.start_run()
            self.params = self.params_cls.suggest(trial)
            self.current_trial = trial
            
            try:
                payload = self._run_training()
                lit_model, best_path, metrics = payload.model, payload.ckpt_path, payload.metrics
            except optuna.exceptions.TrialPruned:
                self.tracker.end_run('killed') 
                raise
            except Exception as e:
                self.tracker.end_run('failed')
                raise e
            
            # Extract target metric
            target_metric_value = metrics.get(f"best_{self.monitor_metric}")
            if target_metric_value is None:
                target_metric_value = metrics.get(self.monitor_metric)
                
            trial.set_user_attr(f"best_{self.monitor_metric}", target_metric_value)
            
            # 2. Check if this specific trial outperformed all previous attempts
            current_best = self._best_value
            is_better = (target_metric_value > current_best) if self.monitor_mode == "max" else (target_metric_value < current_best)
            
            if is_better:
                trial.study.set_user_attr(f"best_{self.monitor_metric}", target_metric_value)
                
                # 3. Instead of uploading immediately, pack and store the current winning configuration
                best_payload = payload
            
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

        # Execute optimization loop
        study.optimize(objective, n_trials=self.n_trials)
        
        # 4. POST-STUDY: Upload ONLY the absolute best model payload to the registry
        if self.registry_cls and best_payload is not None:
            print(f"--> Optimization complete. Uploading the best model from the study to the registry...")
            registry = self.registry_cls()
            registry.upload_model(best_payload)
            
        return study