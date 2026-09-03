# How to Use This Repo

This repository is best understood as a research project with operational tooling.

## Common use cases

- Explore the crowd-counting pipeline and its data assumptions
- Train a model on the supported benchmark datasets
- Compare runs in MLflow
- Understand why prior runs may differ from the current pipeline
- Diagnose issues caused by data mismatch or preprocessing drift

## Recommended reading order

1. [Project overview](03-project-overview.md)
2. [Repository map](05-repo-map.md)
3. [Quickstart](04-quickstart.md)
4. [Dataset contract](../05-data//01-datasets.md)
5. [Data pipeline](../05-data/02-data-pipeline.md)
6. [System overview](../03-architecture/05-system-overview.md)

## Good workflow

1. Verify that the dataset is in the expected folder structure.
2. Check the current preprocessing contract before changing anything.
3. Run the relevant training script.
4. Inspect saved metrics and MLflow metadata.
5. Record important changes in the decisions and history docs.

The `docs/04-decisions/` directory contains the complete ADR record. Do not rewrite an ADR to make a current implementation look cleaner; add a new decision when the project changes direction.
