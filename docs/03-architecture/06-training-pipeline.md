# Training Pipeline

`scripts/train.py` builds a configured `StandardRunner`. The runner creates the datamodule and Lightning model, starts the configured tracker, trains with PyTorch Lightning, evaluates validation and training outputs, logs metrics, and stages a model payload for registry storage when configured. Default callbacks include early stopping, checkpointing, and learning-rate monitoring.

`scripts/tune.py` uses `OptunaTuner`. Each trial receives a `BaseParams` configuration suggested from the Optuna trial and its own tracking context, while the shared runner workflow remains unchanged.

Training uses joint image-density spatial transforms, moderate augmentation, stride-compatible padding, and label scaling. The composite objective combines density reconstruction and structure with explicit attention supervision; later experiments add count-level supervision and carefully scaled weights. Validation can use batching and fixed five-crop processing for large images.

Small preprocessing or loss changes alter the training signal, so dataset layout, split behavior, parameters, and artifacts must be recorded together when comparing runs.
