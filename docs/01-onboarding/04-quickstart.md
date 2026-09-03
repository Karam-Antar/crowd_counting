# Quickstart

## Setup

Create and activate a Python environment, then install the repository dependencies:

```powershell
conda create -n crowd_counting python=3.10
conda activate crowd_counting
pip install -r requirements.txt
```

For serving-specific dependencies, also install `requirements-serve.txt`.

Confirm that dataset paths in `src/config.py` or the environment configuration point to the local data directory. The required layout and image/label ordering contract are documented in [01-datasets.md](../01-datasets.md).

## Training

Run a fixed-configuration experiment with:

```powershell
python scripts/train.py
```

Run the Optuna tuning workflow with:

```powershell
python scripts/tune.py
```

The scripts use the project configuration and dataset contract; inspect those before starting a long run.

## Validation

After training, verify:

- the target map and image are aligned,
- the metrics are sensible for the chosen dataset,
- MLflow metadata reflects the actual run configuration,
- the model artifact is saved to the expected location.

For large images, use the documented batching and five-crop settings only after checking available GPU memory. The project’s current experiments are centered on Shanghai Part A; intensive JHU-Crowd++ tuning is deferred.
