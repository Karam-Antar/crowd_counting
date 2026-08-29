from abc import ABC, abstractmethod
from typing import Optional, Any

from src import config
from src.core.model_wrapper import ModelWrapper
from src.core.params import BaseParams
from src.models.lit_model import BaseLitModel
from src.utils.model_registry.utils import ModelPayload


class BaseRegistry(ABC):
    """Abstract interface for registry-backed model artifact storage strategies.

    Implementations define how a trained model payload is uploaded, downloaded, and
    reloaded from a model registry or artifact store while preserving the experiment
    metadata and parameter bundle.
    """
    
    def __init__(self, experiment_name: Optional[str] = None, tracker: Any = None):
        """Initialize the registry base configuration.

        Args:
            experiment_name (Optional[str]): Name of the target experiment or model family.
            tracker: Logger or tracking client used to stamp uploaded artifacts.
        """
        self.experiment_name = experiment_name
        self.tracker = tracker



    @abstractmethod
    def upload_model(self, payload: ModelPayload):
        """Upload a trained payload to the concrete artifact backend.

        Args:
            payload (ModelPayload): Model, tracker, params, and metrics to persist.

        Returns:
            Any: Backend-specific upload result.
        """
        pass

    @abstractmethod
    def download_model(self, model_name: str, **kwargs) -> tuple[list[str], str]:
        """Download a model artifact and return local path information.

        Args:
            model_name (str): Registered model name or model URI identifier.

        Returns:
            tuple[list[str], str]: Relative file paths and the local download directory.
        """
        pass


        
    @abstractmethod
    def load_model(self, model_name: str, **kwargs):
        """Load a stored model from the backend into the runtime environment.

        Args:
            model_name (str): Model identifier used to resolve the object in storage.

        Returns:
            Any: The backend-specific loaded model object and associated metadata.
        """
        pass
        # downloaded_paths, download_dir = self.download_model(model_name, **kwargs)
        # model_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.pt2')), None)
        # if not model_relative_path:
        #     raise FileNotFoundError("No .pt2 file found in the downloaded model artifacts!")
        
        # model = ExportedModel(load_model(f'{download_dir}/{model_relative_path}'))
        # params = self.load_params(downloaded_paths, download_dir)
        # return model, params
