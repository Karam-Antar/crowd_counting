# Migration Notes

This file records changes to the project that required a migration of expectations, assumptions, or workflow.

Typical examples include:

- new dataset folder conventions,
- changed preprocessing or augmentation behavior,
- updates to config constants,
- changes in logging or training scripts,
- model or artifact compatibility updates.

These notes matter because experiment history often includes code that was valid at one point in time but not at the current one.

## Data and target contract

- [Geometry-adaptive targets](01-adrs/0002-use-geometry-adaptive-targets.md) became the target-generation contract for perspective and scale variation.
- [Invalid or mismatched samples](01-adrs/0003-remove-mislabeled-data.md) were removed before training and evaluation.
- [JHU-Crowd++](01-adrs/0010-unify-the-dir-structure-of-jhu-crowd-pp-with-shanghai.md) was reorganized to follow the Shanghai directory layout, allowing one datamodule to switch datasets through configuration.
- [Negative samples](01-adrs/0023-add-negative-samples-to-shanghai-part-a.md) were injected into the Shanghai Part A pool before the normal train/validation split. Easy negatives included blank or landscape images; hard negatives included birds, cars, and dense non-human patterns. Most were high resolution.
- [Ten negative samples](01-adrs/0027-remove-10-of-the-manually-added-negative-samples.md) were later removed to reduce an observed underestimation bias while preserving negative supervision.

## Model and loss contract

- The project moved to [pretrained encoder-decoder models](01-adrs/0008-switch-to-encoder-decoder-arc.md), replacing earlier architecture experiments.
- The model gained an [attention branch](01-adrs/0016-add-attention-head-at-model-end.md), then explicit mask supervision through [BCE](01-adrs/0017-adjust-loss-function-add-bce.md), followed by [focal loss](01-adrs/0020-replace-bce-with-focal-loss.md) for class imbalance.
- [MSE](01-adrs/0021-replace-mse-with-huber.md) was replaced by Huber loss, and [count-level Huber supervision](01-adrs/0030-try-using-count-loss-with-ssim-and-focal-loss.md) was later combined with SSIM and focal loss.
- Loss weights now require deliberate [scale and penalty calibration](01-adrs/0031-put-more-effort-to-scale-losses-and-hp.md). `huber_delta` was lowered again in [ADR 0032](01-adrs/0032-decrease-huber-delta.md) for more stable optimization.
- [Hard threshold gating](01-adrs/0024-replace-hard-gating-and-manual-threshold-with-soft-gating-in-attention-head.md), such as `sigmoid_output > 0.88`, was replaced by direct sigmoid soft gating and a final convolution, removing per-run threshold selection.

## Evaluation and serving workflow

- [Positive NAE and MBE](01-adrs/0026-add-custom-metrics-mbe-and-positive-nae.md) were added to avoid unstable zero-denominator normalization and expose systematic over- or underestimation.
- Validation and test loaders gained [batching with padding](01-adrs/0025-add-batching-ability-to-validation-and-test-dataloaders.md); batch sizes must account for the largest padded image and dimensions must remain compatible with the network.
- Large-image validation gained [five fixed crops](01-adrs/0028-add-five-crops-for-validation.md) to reduce memory pressure while keeping repeated evaluations comparable.
- Model serving migrated to an [MLflow PyFunc wrapper](01-adrs/0012-give-up-script-and-export-logging-formats-for-pyfunc.md) that packages parameters, weights, preprocessing, postprocessing, and source code together.

## Configuration and experiment workflow

- [BaseParams](01-adrs/0005-let-every-training-testing-component-rely-on-params-object.md) became the shared configuration object for data, architecture, optimization, loss, tuning, and inference. Parameters are serialized with artifacts so training and serving use the same contract.
- [Runner inheritance and tracker/registry interfaces](01-adrs/0004-build-inheritance-arc-for-experiment-tracking.md) centralized shared training behavior while keeping standard runs and Optuna trials separate.
- JHU-Crowd++ training remains [deferred](01-adrs/0011-try-training-on-jhu-crowd-pp-dataset.md) for intensive tuning because its high-resolution data makes each experiment substantially more expensive.
