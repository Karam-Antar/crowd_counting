from abc import ABC, abstractmethod
import datetime
from typing import Dict, Any, Optional
from lightning.pytorch.loggers import Logger


class BaseTracker(ABC):
    """Abstract interface for experiment-tracking strategies used by training runs.

    Concrete implementations provide the logger object and lifecycle methods for
    creating, updating, and closing the experiment context used by Lightning and MLflow.
    """
    def __init__(self, experiment: str, run_name: str | list[str] | None = None):
        """Initialize the tracking strategy with experiment metadata.

        Args:
            experiment (str): Experiment name used by the tracking backend.
            run_name (str | list[str] | None): Optional run identifier or nested run names.
        """
        self.experiment = experiment
        timestamp = lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{timestamp()}"
        self.nested = isinstance(run_name, list)
        if self.nested:
            run_name = [e if e else str(timestamp()) for e in run_name]
        self.logger: Logger | None = None
    
    @property
    def model_name(self):
        """Return the model name used in the tracking backend."""
        return 'model'
    
    @property
    def full_experiment_name(self):
        """Return the fully qualified experiment identifier for metadata."""
        return 'experiment'
    
    @abstractmethod
    def get_logger(self) -> Logger:
        """Return the backend-specific logger used by the Lightning trainer."""
        pass

    @abstractmethod
    def start_run(self):
        """Initialize the run or trial context for the selected backend."""
        pass

    @abstractmethod
    def log_init(self, params: dict):
        """Log the initial run configuration and experiment metadata."""
        pass

    @abstractmethod
    def log_results(self, metrics: dict, params: dict):
        """Log the final evaluated metrics and hyperparameters."""
        pass

    @abstractmethod
    def end_run(self, status: str = 'success'):
        """Close the run or mark it as failed/successful."""
        pass
