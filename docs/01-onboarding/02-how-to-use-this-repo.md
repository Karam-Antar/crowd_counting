# How to Use This Repo

This repository is best understood as a research project with operational tooling.

## Common use cases

- Explore the crowd-counting pipeline and its data assumptions
- Train a model on the supported benchmark datasets
- Compare runs in MLflow
- Understand why prior runs may differ from the current pipeline
- Diagnose issues caused by data mismatch or preprocessing drift

## Recommended reading order

1. [project-overview.md](03-project-overview.md)
2. [repo-map.md](05-repo-map.md)
3. [quickstart.md](04-quickstart.md)
4. [data/data_pipeline.md](../04-data/01-data-pipeline.md)
5. [data/datasets.md](../04-data/04-datasets.md)
6. [architecture/system-overview.md](../03-architecture/05-system-overview.md)

## Good workflow

1. Verify that the dataset is in the expected folder structure.
2. Check the current preprocessing contract before changing anything.
3. Run the relevant training script.
4. Inspect saved metrics and MLflow metadata.
5. Record important changes in the decisions and history docs.
