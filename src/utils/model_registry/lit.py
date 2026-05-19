import tempfile
from pathlib import Path
import litmodels

from src import config
from src.utils.model_persistence import save_model
from src.utils.model_registry.base import BaseRegistry
from src.utils.model_registry.utils import ModelPayload, prepare_temp_dir


class LitRegistry(BaseRegistry):

    def upload_model(self, payload: ModelPayload):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_path = Path(tmp_dir)
            prepare_temp_dir(artifacts_path, payload, payload.tracker.full_experiment_name)
            save_model(payload.model, payload.params, model_id='model', model_path=artifacts_path)
            lit_exp = self.tracker.active_experiment 
            model_url = f'{lit_exp.teamspace.owner.name}/{lit_exp.teamspace.name}/{payload.tracker.model_name}'
            model_info = litmodels.upload_model_files(name=model_url, path=artifacts_path)
            lit_exp.log_metadata({'model_registry': f'{model_url}:{model_info.version}'})

    def download_model(self, model_name: str, **kwargs) -> tuple[list[str], str]:
        model_suffix = model_name.rsplit('/', 1)[-1]
        download_dir = f"{config.MODEL_SAVE_PATH}/{model_suffix}"
        downloaded_paths = litmodels.download_model(model_name, download_dir=download_dir)
        return downloaded_paths, download_dir
