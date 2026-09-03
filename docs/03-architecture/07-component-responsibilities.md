# Component Responsibilities

This page maps the current source tree to the responsibilities described in the project documentation. It is a navigation aid, not a second implementation specification.

## Core configuration and inference

- `src/config.py`: project-wide constants, paths, environment-backed settings, and defaults.
- `src/core/params.py`: `BaseParams`, the shared configuration object for data, architecture, training, losses, tuning, serialization, and inference.
- `src/core/inference.py`: inference entry points and prediction flow.
- `src/core/model_wrapper.py`: model-loading and prediction wrapper behavior.
- `src/core/pyfunc.py`: MLflow PyFunc serving wrapper that reconstructs the model from parameters and weights.

## Data

- `src/data/dataset.py`: image and density-map dataset loading, including the ordered image/label pairing contract.
- `src/data/transform.py`: image and density-map transforms, normalization, scaling, padding, and crop-related processing.
- `src/data/transform_sample.py`: sample-level transformation support.
- `src/data/datamodule.py`: dataset setup, train/validation/test construction, dataloaders, batching, and collate functions.

The complete preprocessing behavior is documented in [Data Pipeline](../05-data/02-data-pipeline.md), and the dataset layout and label contract are documented in [Datasets](../05-data/01-datasets.md).

## Models and objectives

- `src/models/encoder_decoder.py`: encoder-decoder construction and decoder behavior.
- `src/models/model.py`: crowd-counter model assembly.
- `src/models/lit_model.py`: PyTorch Lightning module coordinating forward passes, loss computation, optimization, and logged metrics.
- `src/models/loss.py`: density, structural, attention-mask, and count-related loss components.
- `src/models/metrics.py`: evaluation metrics used by training and validation.

The architectural evolution, including the attention head, soft gating, focal loss, Huber loss, and count supervision, is recorded in [Architecture Decisions](../04-decisions/01-decision-log.md) and the linked ADRs.

## Experiment orchestration

- `src/experiment/base.py`: shared experiment-runner workflow.
- `src/experiment/standard.py`: one fixed-parameter training run.
- `src/experiment/optuna_tuner.py`: Optuna trial creation and tuning orchestration.

The runner hierarchy keeps common model construction, callbacks, evaluation, and payload handling in one place while allowing standard runs and tuning trials to differ in orchestration.

## Tracking and registry

- `src/utils/experiment_trackers/base.py`: tracker interface.
- `src/utils/experiment_trackers/mlflow.py`: MLflow tracker implementation.
- `src/utils/model_registry/base.py`: model-registry interface.
- `src/utils/model_registry/mlflow.py`: MLflow registry implementation.
- `src/utils/model_registry/utils.py`: registry support utilities.
- `src/utils/registries.py`: registry-related helpers.

Tracking records parameters and results; registry code handles model payload storage when configured. Runtime artifacts are not assumed to be present in the source repository.

## Supporting utilities

- `src/utils/helpers.py`: shared helper functions.
- `src/utils/inspection.py`: inspection and diagnostic helpers.
- `src/utils/visualization.py`: visualization support for model and data inspection.

## Main execution relationship

```text
BaseParams
    |
    +--> CrowdDataModule --> datasets and transforms
    |
    +--> CrowdCounter --> encoder-decoder and attention path
    |
    +--> BaseLitModel --> losses, metrics, and optimization
    |
    +--> Experiment runners --> Lightning training and evaluation
    |
    +--> Tracker / registry --> run records and model payloads
    |
    +--> Inference / PyFunc --> reproducible serving
```
