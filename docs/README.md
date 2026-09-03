# Crowd Counting Project Documentation

This documentation describes the verified project contract and the decisions that shaped it. It intentionally avoids publishing unsupported run IDs, dates, production claims, or metric tables that are not present in the repository.

## Start here

1. [Project overview](01-onboarding/03-project-overview.md)
2. [Quickstart](01-onboarding/04-quickstart.md)
3. [Dataset contract](05-data/01-datasets.md)
4. [Data pipeline](05-data/02-data-pipeline.md)
5. [Architecture](03-architecture/05-system-overview.md)
6. [Decision log](04-decisions/01-decision-log.md)

## Documentation map

- [Onboarding](01-onboarding): repository orientation and contribution workflow
- [Context](02-context): problem, goals, constraints, and domain background
- [Architecture](03-architecture): configuration, data flow, model, inference, and training design
- [Datasets](05-data/01-datasets.md): supported datasets, layouts, labels, and acquisition notes
- [Data pipeline](05-data/02-data-pipeline.md): implemented loading, transforms, padding, batching, and five-crop evaluation
- [Decisions](04-decisions): chronological decision log, ADRs, migrations, and rationale for removed approaches

## Scope note

The decision records are the project’s experiment history. Dataset and preprocessing behavior are defined by the two root data documents. Source code and scripts remain authoritative when implementation details differ from historical descriptions.
