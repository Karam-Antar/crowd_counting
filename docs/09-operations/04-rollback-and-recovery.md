# Rollback and Recovery

This file documents how to recover a stable state after a bad experiment or a broken configuration change.

## Recovery workflow

- identify the last known-good configuration,
- compare it against the current code and dataset state,
- revert the specific component that introduced the issue,
- validate the dataset and preprocessing contract before retraining,
- keep enough notes so the issue is understandable for the next person.

This is especially important in projects with many different run families and evolving experiment logic.
