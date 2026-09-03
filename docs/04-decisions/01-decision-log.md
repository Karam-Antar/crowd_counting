# Decision Log

This file documents major technical choices that shaped the project.

The project evolved from architecture selection into loss-function engineering and data- and evaluation-pipeline refinement. The current path is the result of the accepted decisions below; rejected and deferred experiments are summarized in the companion decision pages.

## Chronological decisions

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](01-adrs/0001-use-density-map-regression.md) | Use density-map regression so predictions retain spatial information and counts can be obtained by integrating the map. | Accepted |
| [0002](01-adrs/0002-use-geometry-adaptive-targets.md) | Generate geometry-aware density targets to handle perspective and scale variation. | Accepted |
| [0003](01-adrs/0003-remove-mislabeled-data.md) | Remove invalid or mismatched image-density pairs instead of training on corrupted supervision. | Accepted |
| [0004](01-adrs/0004-build-inheritance-arc-for-experiment-tracking.md) | Share training workflow through `BaseExperimentRunner`, with specialized standard-run and Optuna runners plus tracker and registry interfaces. | Accepted |
| [0005](01-adrs/0005-let-every-training-testing-component-rely-on-params-object.md) | Make `BaseParams` the serialized source of truth for data, model, training, loss, tuning, and inference configuration. | Accepted |
| [0006](01-adrs/0006-give-up-coord-att.md) | Remove coordinate attention because its cost did not produce measurable validation gains. | Superseded |
| [0007](01-adrs/0007-give-up-hrnet.md) | Stop using HRNet as the primary backbone after its performance plateau and high resource cost. | Superseded |
| [0008](01-adrs/0008-switch-to-encoder-decoder-arc.md) | Adopt pretrained encoder-decoder models with skip connections as the main dense-prediction architecture. | Accepted |
| [0009](01-adrs/0009-stop-trying-to-fix-heavy-augmentation.md) | Keep augmentation moderate because heavier augmentation added cost, artifacts, and instability without reliable gains. | Accepted |
| [0010](01-adrs/0010-unify-the-dir-structure-of-jhu-crowd-pp-with-shanghai.md) | Reshape JHU-Crowd++ to the Shanghai layout so one datamodule can load both datasets. | Accepted |
| [0011](01-adrs/0011-try-training-on-jhu-crowd-pp-dataset.md) | Try JHU-Crowd++ but defer intensive tuning until Shanghai Part A is optimized. | Deferred |
| [0012](01-adrs/0012-give-up-script-and-export-logging-formats-for-pyfunc.md) | Use an MLflow PyFunc wrapper with checkpoint weights, serialized parameters, and packaged source code for serving. | Accepted |
| [0013](01-adrs/0013-add-ssim-loss.md) | Add SSIM to density loss to preserve spatial structure alongside magnitude accuracy. | Accepted |
| [0014](01-adrs/0014-try-sliding-window-patching-prediction.md) | Reject sliding-window inference because independent patches lose global perspective and density context. | Rejected |
| [0015](01-adrs/0015-try-ensembled-model-prediction.md) | Reject model ensembles because memory, latency, and deployment complexity outweighed their benefit. | Rejected |
| [0016](01-adrs/0016-add-attention-head-at-model-end.md) | Add an attention head to learn foreground/background confidence and reduce false positives. | Accepted with caveat |
| [0017](01-adrs/0017-adjust-loss-function-add-bce.md) | Add explicit binary-mask supervision with BCE as the first attention-head loss. | Accepted experimentally |
| [0018](01-adrs/0018-add-count-penalty-loss.md) | Explore direct count-error supervision, but reject the initial formulation after degraded spatial predictions and metrics. | Rejected |
| [0019](01-adrs/0019-add-spatially-weighted-loss.md) | Explore spatially weighted density loss, but reject it because indirect weighting did not solve false positives. | Rejected |
| [0020](01-adrs/0020-replace-bce-with-focal-loss.md) | Replace BCE with focal loss so attention training focuses on hard negatives under severe class imbalance. | Accepted |
| [0021](01-adrs/0021-replace-mse-with-huber.md) | Replace MSE with Huber loss to limit outlier influence in label-scaled density maps. | Accepted |
| [0022](01-adrs/0022-try-dynamic-loss-weightning.md) | Explore learned or adaptive loss weights, but defer the direction because optimization became less stable without clear gains. | Deferred |
| [0023](01-adrs/0023-add-negative-samples-to-shanghai-part-a.md) | Add 30–40 high-resolution easy and hard negative images to give the attention head explicit non-crowd examples. | Accepted |
| [0024](01-adrs/0024-replace-hard-gating-and-manual-threshold-with-soft-gating-in-attention-head.md) | Replace thresholded binary gating with sigmoid-valued soft gating followed by a final convolution. | Accepted |
| [0025](01-adrs/0025-add-batching-ability-to-validation-and-test-dataloaders.md) | Batch validation and test samples using padding to batch maxima and multiples of 32. | Accepted |
| [0026](01-adrs/0026-add-custom-metrics-mbe-and-positive-nae.md) | Add Positive NAE for numerical stability and MBE for directional prediction bias. | Accepted |
| [0027](01-adrs/0027-remove-10-of-the-manually-added-negative-samples.md) | Remove 10 negative samples to reduce underestimation while retaining negative supervision. | Accepted |
| [0028](01-adrs/0028-add-five-crops-for-validation.md) | Use five fixed crops for large-image validation to reduce memory and runtime while keeping comparisons consistent. | Accepted |
| [0029](01-adrs/0029-preload-data-into-memory-when-training.md) | Test preloading and pretransformation, then reject it as an ineffective primary optimization. | Rejected / Deferred |
| [0030](01-adrs/0030-try-using-count-loss-with-ssim-and-focal-loss.md) | Add overall-count Huber loss while retaining SSIM and focal loss to stabilize MBE and constrain aggregate count. | Accepted |
| [0031](01-adrs/0031-put-more-effort-to-scale-losses-and-hp.md) | Separate numerical scale-unification factors from penalty factors when weighting loss terms. | Accepted |
| [0032](01-adrs/0032-decrease-huber-delta.md) | Lower Huber delta so the loss enters its stable linear regime sooner for label-scaled errors. | Accepted |

## Current direction

The current design combines a full-image encoder-decoder with an attention branch, soft gating, SSIM, focal loss, and Huber-based count supervision. Training configuration is centralized in `BaseParams`, experiments are tracked through the runner/tracker abstractions, and validation uses batching and fixed crops where image size requires it. Shanghai Part A remains the main tuning dataset; JHU-Crowd++ is available for broader validation and later fine-tuning.
