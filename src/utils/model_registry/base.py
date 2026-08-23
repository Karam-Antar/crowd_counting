from abc import ABC, abstractmethod
from typing import Optional, Any

from src import config
from src.core.exported_model import ExportedModel
from src.core.params import BaseParams
from src.models.lit_model import BaseLitModel
from src.utils.model_registry.utils import ModelPayload


class BaseRegistry(ABC):
    """Abstract interface for model registry strategies."""
    
    def __init__(self, experiment_name: Optional[str] = None, tracker: Any = None):
        self.experiment_name = experiment_name
        self.tracker = tracker



    @abstractmethod
    def upload_model(self, payload: ModelPayload):
        """Uploads the formatted payload directory to the specific artifact store."""
        pass

    @abstractmethod
    def download_model(self, model_name: str, **kwargs) -> tuple[list[str], str]:
        """Downloads the model directory and returns (list_of_relative_paths, local_download_dir)."""
        pass


        
    @abstractmethod
    def load_model(self, model_name: str, **kwargs):
        pass
        # downloaded_paths, download_dir = self.download_model(model_name, **kwargs)
        # model_relative_path = next((p for p in downloaded_paths if p.casefold().endswith('.pt2')), None)
        # if not model_relative_path:
        #     raise FileNotFoundError("No .pt2 file found in the downloaded model artifacts!")
        
        # model = ExportedModel(load_model(f'{download_dir}/{model_relative_path}'))
        # params = self.load_params(downloaded_paths, download_dir)
        # return model, params
