# ADR 0003: Add negative samples

## Status

Accepted

## Context

Some data points represented scenes without people or with very low crowd presence. These underrepresented examples could bias training if they were not explicitly included.

## Decision

Add negative samples to improve dataset diversity and discourage over-prediction in low-crowd scenes.

## Consequences

- more balanced training distribution,
- better generalization in sparse scenes,
- stronger need to document the data changes clearly.
