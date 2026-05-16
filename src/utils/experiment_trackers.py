from abc import ABC, abstractmethod
import datetime
from typing import Dict, Any, Optional
from lightning.pytorch.loggers import Logger, LitLogger, MLFlowLogger
import mlflow
import litlogger

class BaseTracker(ABC):
    """Abstract interface for experiment tracking strategies."""
    def __init__(self, experiment: str, run_name: str | list[str] | None = None):
        self.experiment = experiment
        timestamp = lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{timestamp()}"
        self.nested = isinstance(run_name, list)
        if self.nested:
            run_name = [e if e else str(timestamp()) for e in run_name]
        # self.model_name = 'model'
        # self.full_experiment_name = 'experiment'
    
    @abstractmethod
    def get_logger(self) -> Logger:
        """Returns the specific PyTorch Lightning Logger instance."""
        pass

    @abstractmethod
    def start_run(self):
        """Initializes a run/trial context."""
        pass

    @abstractmethod
    def log_results(self, metrics: dict, params: dict):
        """Logs the final evaluated metrics and hyperparameters."""
        pass

    @abstractmethod
    def end_run(self, status: str = 'success'):
        """Closes the current run/trial context."""
        pass


class LitTracker(BaseTracker):
    """Concrete strategy for LitLogger."""
    def __init__(self, experiment: str, run_name: str | list[str] | None = None):
        super().__init__(experiment, run_name)
        self.full_run_name = '/'.join(self.run_name) if self.nested else run_name
        self.active_experiment = None
        self.current_name = None

    def get_logger(self) -> Logger:
        self.current_name = f'{self.experiment}/{self.full_run_name}'
        self.logger = LitLogger(name=self.current_name, save_logs=False)
        return self.logger

    def start_run(self):
        # LitLogger handles runs dynamically via init(), no strict pre-start needed
        pass

    def log_results(self, metrics: dict, params: dict):
        # litlogger.init needs to be called after training to log final dicts
        self.active_experiment = litlogger.init(name=self.current_name)
        self.active_experiment.log_metrics(metrics)
        self.active_experiment.log_metadata(params)

    def end_run(self, status: str = 'success'):
        if self.active_experiment:
            self.active_experiment.finalize(status)
            self.active_experiment = None
    
    @property
    def model_name(self) -> str:
        return self.full_run_name.replace('/', '_')

    @property
    def full_experiment_name(self) -> str:
        return f'{self.experiment}/{self.full_run_name}'


class MLFlowTracker(BaseTracker):
    """Concrete strategy for MLFlow."""
    def __init__(self, experiment: str, run_name: str | list[str] | None = None):
        super().__init__(experiment, run_name)
        self.base_run = None
        self.sub_run = None
        self.logger_name = self.run_name
        if self.nested:
            self.base_run, self.sub_run = self.run_name
            self.logger_name = self.sub_run


    def get_logger(self) -> Logger:
        run_tags = {}
        if self.base_run_id:
            run_tags["mlflow.parentRunId"] = self.base_run_id
        self.logger = MLFlowLogger(experiment_name=self.experiment, run_name=self.logger_name, tags=run_tags, synchronous=False)
        return self.logger

    def start_run(self):
        mlflow.set_experiment(self.experiment)
        if self.base_run:
            self.base_run_id = mlflow.start_run(run_name=self.base_run, run_id=self._get_existing_run_id(self.base_run)).info.run_id

    def log_results(self, metrics: dict, params: dict):
        self.logger.log_metrics(metrics)
        # mlflow.log_params(params)

    def end_run(self, status: str = 'success'):
        if self.base_run:
            mlflow.end_run()
    
    @property
    def model_name(self) -> str:
        return f'{self.base_run}_{self.sub_run}' if self.nested else self.logger_name
    
    @property
    def full_experiment_name(self) -> str:
        return f'{self.experiment}/{self.model_name.replace('_', '/')}'

    def _get_existing_run_id(self, run_name: str) -> Optional[str]:
        runs = mlflow.search_runs(
            experiment_names=[self.experiment],
            filter_string=f"attributes.run_name = '{run_name}'",
            max_results=1
        )
        return runs.iloc[0].run_id if not runs.empty else None