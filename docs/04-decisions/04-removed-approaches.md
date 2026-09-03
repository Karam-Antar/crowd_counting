# Removed Approaches

This file documents ideas and pipeline choices that were tested and later removed from the primary path.

Examples may include:

- alternative augmentation strategies,
- older fold or split logic,
- earlier training scripts with different assumptions,
- architecture directions that were superseded.

The goal here is to preserve the reason for removal so that future teams do not have to rediscover it from scratch.

## Architecture and inference

- **Coordinate attention** was removed after adding compute without measurable gains ([ADR 0006](01-adrs/0006-give-up-coord-att.md)).
- **HRNet** was removed as the primary backbone after a performance plateau and high resource requirements ([ADR 0007](01-adrs/0007-give-up-hrnet.md)).
- **Sliding-window inference** was removed because patches lost the global context needed for perspective and density reasoning ([ADR 0014](01-adrs/0014-try-sliding-window-patching-prediction.md)).
- **Ensembled inference** was removed because repeated model execution made memory, latency, and deployment costs too high ([ADR 0015](01-adrs/0015-try-ensembled-model-prediction.md)).
- **Hard attention gating** and a manually selected threshold were removed. The current path multiplies the density map by continuous sigmoid values and applies a final convolution ([ADR 0024](01-adrs/0024-replace-hard-gating-and-manual-threshold-with-soft-gating-in-attention-head.md)).

## Training and data pipeline

- **Heavy augmentation** was reduced and is no longer treated as the main generalization strategy because it was slower, sometimes unstable, and not reliably better ([ADR 0009](01-adrs/0009-stop-trying-to-fix-heavy-augmentation.md)).
- **Dataset preloading and pretransformation** were tested and removed as a primary optimization after negligible or inconsistent throughput gains ([ADR 0029](01-adrs/0029-preload-data-into-memory-when-training.md)).
- **Ten manually added negative samples** were removed to correct underestimation bias; the remaining negatives continue to provide empty-scene supervision ([ADR 0027](01-adrs/0027-remove-10-of-the-manually-added-negative-samples.md)).

## Loss and serving implementations

- **Script and graph-export logging** were removed in favor of PyFunc because TorchScript and export formats were brittle under architecture and dependency changes ([ADR 0012](01-adrs/0012-give-up-script-and-export-logging-formats-for-pyfunc.md)).
- **BCE mask loss** was superseded by focal loss. BCE established explicit attention supervision, but its treatment of the highly imbalanced mask left hard negatives under-trained ([ADR 0017](01-adrs/0017-adjust-loss-function-add-bce.md), [ADR 0020](01-adrs/0020-replace-bce-with-focal-loss.md)).
- **Point-wise MSE density loss** was superseded by Huber loss because large label-scaled errors dominated optimization ([ADR 0021](01-adrs/0021-replace-mse-with-huber.md)).
- **The initial count-penalty formulation** was removed after it degraded spatial density maps; the later count-level Huber formulation retained SSIM and focal supervision instead ([ADR 0018](01-adrs/0018-add-count-penalty-loss.md), [ADR 0030](01-adrs/0030-try-using-count-loss-with-ssim-and-focal-loss.md)).
- **Spatially weighted loss** was removed because indirect error weighting did not provide the explicit foreground/background signal needed to suppress false positives ([ADR 0019](01-adrs/0019-add-spatially-weighted-loss.md)).
