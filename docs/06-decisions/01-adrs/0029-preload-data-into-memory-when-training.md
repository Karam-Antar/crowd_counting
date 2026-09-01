# ADR 0029: Avoid optimizing training by preloading dataset into memory

Rejected / Deferred

## Context

During training on the JHU-Crowd++ dataset, we observed noticeable latency in the validation loop. The initial hypothesis was that the GPU was mostly idling while waiting for data to be loaded from disk or transformed on the fly, especially during validation passes where the same dataset is iterated repeatedly.

To test this, we investigated whether preloading the training and/or validation data into RAM or VRAM could reduce the apparent bottleneck. We also considered a more aggressive variant: caching or pretransforming the dataset ahead of time so that the model would receive already-prepared tensors rather than doing work in the data pipeline during training.

## Decision

We tried to preload, and in some cases even pretransform, the training and validation data into memory to reduce data-access delays and improve throughput during training. This included keeping batches or full dataset shards resident in RAM and, where practical, preparing transformed inputs ahead of time so the training loop would not have to perform repeated work during the validation phase.

## Why this was not kept

This did not introduce any clear improvement in end-to-end training speed or validation throughput. The gains were either negligible or inconsistent, and they did not meaningfully offset the added complexity and memory overhead.

This suggests that the bottleneck is likely not primarily that the GPU sits idle waiting for data to arrive. Instead, the more likely explanation is that the validation pass itself is expensive because of the size of the dataset, the size and complexity of the model architecture, and the cost of running full forward passes over large input tensors and density-estimation outputs. In other words, the time is being spent in model inference and computation rather than in data loading or preprocessing.

## Consequences

- We do not treat dataset preloading or full pretransformation as a primary optimization for this training setup.
- We should continue focusing optimization effort on model efficiency, validation cadence, inference cost, architecture-level bottlenecks or techniques similar to Five Crops instead of expanding memory-heavy data caching strategies.
- The data pipeline remains simple and maintainable, without adding large RAM/VRAM residency requirements that may be hard to scale across environments.
- If future experiments show a different bottleneck under new dataset sizes or hardware, revisiting preloading or precomputation may still be worthwhile, but it should be justified with measured performance evidence rather than by a generic assumption about data latency.

## Summary

The attempted data preloading strategy was informative but not effective. It did not provide a convincing improvement, which points away from the idea that validation latency is mostly caused by waiting for data and toward the conclusion that the expensive part is the model inference itself under the current dataset and architecture conditions.
