from typing import Optional
from lightning.pytorch.loggers import Logger, MLFlowLogger
import mlflow

from src.core.params import BaseParams
from src.utils.experiment_trackers.base import BaseTracker


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
    

    def log_init(self, params: dict):
        pass

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
