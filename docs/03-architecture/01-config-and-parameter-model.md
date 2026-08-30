# Config and Parameter Model

The project configuration is centralized in src/config.py. This file contains path configuration, dataset roots, project-wide constants, logging defaults, and reproducibility settings.

Important values include:

- seed values,
- label-scaling factor,
- dataset root paths,
- model artifact locations,
- MLflow and log directory configuration.

The project depends on these settings being stable enough to support meaningful experiment comparison.
