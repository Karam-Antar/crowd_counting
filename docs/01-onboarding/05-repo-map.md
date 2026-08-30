# Repository Map

## Top-level folders

- src/: source code for data loading, model logic, experiment logic, and inference
- scripts/: dataset conversion, training, cleanup, and tuning entry points
- notebooks/: exploratory analysis and notebooks for investigation
- docs/: project knowledge, architecture, and experiment history
- trained_models/: saved checkpoints and model artifacts
- logs/: runtime and training logs

## Most important source areas

- src/data/: dataset, transforms, batching, and datamodule setup
- src/models/: loss, metrics, model wrappers, and training model code
- src/experiment/: experiment orchestration and tuning logic
- src/core/: inference and serving behavior
- src/config.py: global config, data paths, and fixed project constants

## Why the repo matters

The codebase mixes ML engineering work with research exploration, so source structure alone does not tell the whole story. The docs are where the project memory lives.
