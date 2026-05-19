from lightning.pytorch.loggers import Logger, LitLogger
import litlogger

from src.core.params import BaseParams
from src.utils.experiment_trackers.base import BaseTracker


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

    def log_init(self, params: dict):
        self.logger.log_hyperparams(params)

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
