# ADR 0010: Unify the directory structure of JHU-Crowd++ with Shanghai

## Status

Accepted

## Context

The project initially supported two crowd-counting datasets with different directory layouts:
- **Shanghai Part A**: One folder structure convention for images and labels
- **JHU-Crowd++**: A different folder structure with its own organizational scheme

Supporting both layouts required either:
1. Writing separate datamodule implementations, or
2. Unifying the directory structure so a single datamodule can load both datasets

## Decision

Unify the directory structure of JHU-Crowd++ to match Shanghai's layout, eliminating the need for dataset-specific datamodule variants. This allows `CrowdDataModule` to load either dataset transparently.

## Consequences

- **Reduced code duplication**: A single datamodule handles both datasets, reducing maintenance burden.
- **Cleaner dataset abstraction**: The same preprocessing pipeline applies uniformly.
- **Easier dataset switching**: Toggling between datasets requires only changing the `DATASET_PATH` configuration.

## Limitations & Future Work

This approach is pragmatic for two datasets but does not scale. A project with more than two datasets with varied structures needs a more principled approach:
- A dataset registry that maps dataset names to custom loaders
- A plugin-based system where each dataset defines its own schema
- Metadata files (JSON, YAML) that describe dataset structure

For the scope of this project (two primary datasets), unified directory structure was sufficient. Future growth should reconsider this design.

## Related Decisions

- **ADR 0011** (Try training on JHU-Crowd++): The unified structure enabled quick experimentation on the second dataset.
