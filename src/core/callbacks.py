"""PyTorch Lightning callbacks for training."""

import time
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class ReseedCallback(Callback):
    """Callback to reseed random number generators periodically during training.

    This ensures that data augmentation operations get different random seeds
    on each epoch to improve stochasticity.
    """

    def __init__(self, interval=1):
        """
        Args:
            interval: Reseed every N epochs (default: 1, i.e., every epoch).
        """
        super().__init__()
        self.interval = interval

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if trainer.current_epoch % self.interval == 0:
            # Generate a new seed based on current time
            seed = int(time.time()) % (2 ** 32 - 1)
            pl.seed_everything(seed, workers=True, verbose=False)