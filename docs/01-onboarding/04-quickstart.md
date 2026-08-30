# Quickstart

## Setup

1. Create the project environment using the repository requirements.
2. Confirm dataset access and local path configuration.
3. Validate that the expected folder structure exists for the benchmark data.

## Training

Use the scripts in the scripts directory to train or tune the model. Training is highly dependent on the current preprocessing assumptions and dataset layout, so it is best to verify those first.

## Validation

After training, verify:

- the target map and image are aligned,
- the metrics are sensible for the chosen dataset,
- MLflow metadata reflects the actual run configuration,
- the model artifact is saved to the expected location.
