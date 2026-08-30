# ADR 0004: Remove mislabeled data

## Status

Accepted

## Context

Some records were inconsistent, mislabeled, or mismatched with the expected image-density pairing.

## Decision

Remove invalid or inconsistent samples instead of allowing them to corrupt training and evaluation.

## Consequences

- cleaner supervision,
- more meaningful metrics,
- better data contract integrity for future work.
