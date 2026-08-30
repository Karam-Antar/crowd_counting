# Contributor Guide

When contributing to this project, keep in mind that it is an evolving ML research codebase with a long experiment history.

## Rules of thumb

- do not change the data contract without checking the dataset docs,
- verify whether a run is comparable to previous runs before changing training assumptions,
- preserve the reason for changes in the decisions or history documents,
- keep the current baseline stable while experimenting separately.

## Recommended workflow

1. Understand the current data and training contract.
2. Review relevant experiment history.
3. Make the smallest focused change.
4. Validate training and metrics.
5. Record what changed and why.
