from abc import ABC, abstractmethod
import datetime
from typing import Dict, Any, Optional
from lightning.pytorch.loggers import Logger


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
    def log_init(self, params: dict):
        pass

    @abstractmethod
    def log_results(self, metrics: dict, params: dict):
        """Logs the final evaluated metrics and hyperparameters."""
        pass

    @abstractmethod
    def end_run(self, status: str = 'success'):
        """Closes the current run/trial context."""
        pass
