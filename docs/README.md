# Crowd Counting Project Documentation

This documentation set captures the project’s current state, technical context, experiment history, and design rationale.

## Documentation map

- [01-onboarding](01-onboarding): repository overview, quick start, and contributor guidance
- [02-context](02-context): problem framing, project goals, constraints, and domain background
- [03-architecture](03-architecture): system design, data flow, and model/training pipeline
- [04-data](04-data): dataset definitions, data processing, and known data issues
- [05-experiments](05-experiments): MLflow run tracking and experiment comparison guidance
- [06-decisions](06-decisions): ADRs and notes about removed or superseded ideas
- [07-training](07-training): training workflow, tuning, reproducibility, and metrics
- [08-deployment](08-deployment): serving, inference, and production checklist
- [09-operations](09-operations): troubleshooting, scripts, setup, and maintenance notes
- [10-history](10-history): project timeline, changes, and lessons learned
- [11-reference](11-reference): glossary, commands, config keys, and artifact overview
- [12-assets](12-assets): diagrams and supporting visuals

## Purpose

This project is a crowd-counting system built around density-map regression. It has gone through multiple experimental directions, and the documentation exists to preserve the current baseline along with the historical reasoning behind the changes.

## How to read these docs

Start with the onboarding section, then move to data and architecture to understand the real operating contract. If you are debugging a run or trying to understand why something was changed, read the decisions and history sections.
