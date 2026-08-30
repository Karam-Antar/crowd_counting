# Reproducibility

Reproducibility in this project depends on more than a fixed random seed.

The project should track:

- dataset version,
- script or code revision,
- split strategy,
- preprocessing settings,
- label scaling,
- artifact metadata.

Without these details, it becomes difficult to compare older MLflow runs to the current pipeline with confidence.
