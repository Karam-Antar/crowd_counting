# Debugging Training

Training issues are often caused by data contract problems rather than model logic problems.

## Common checks

- confirm image and density target pairing,
- verify density-map scaling,
- inspect transform ordering,
- ensure the correct split logic is used,
- compare the current run with the most relevant successful experiment.

A good debugging workflow in this project starts with the data contract before moving on to neural-network-level debugging.
