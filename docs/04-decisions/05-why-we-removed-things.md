# Why We Removed Things

Some design changes were removed because they were unstable, incompatible with the project contract, or inconsistent with the project’s current data assumptions.

This document is meant to preserve the reasoning, not to judge past attempts. Long-lived ML projects often need to keep a written record of what was abandoned and why.

## The governing principle

An approach was removed when it failed against the project’s real constraints: spatially accurate density maps, reliable total counts, false-positive suppression, stable optimization, practical GPU use, and maintainable serving. The ADR history shows that a theoretically attractive method was not enough; it had to improve the measured behavior of the complete pipeline.

## Why architecture complexity was reduced

Coordinate attention and HRNet were both plausible attempts to preserve or improve spatial detail. They were removed because their extra computation did not translate into better validation performance. The encoder-decoder baseline with pretrained backbones provided a larger immediate gain and a more flexible foundation ([ADR 0006](01-adrs/0006-give-up-coord-att.md), [ADR 0007](01-adrs/0007-give-up-hrnet.md), [ADR 0008](01-adrs/0008-switch-to-encoder-decoder-arc.md)).

Patch-based prediction was removed for a different reason: it broke the task’s information requirements. Crowd density depends on perspective, scale, gradients, and scene-wide context, so isolated patches produced worse predictions ([ADR 0014](01-adrs/0014-try-sliding-window-patching-prediction.md)). Ensembles could improve accuracy, but their linear memory and latency cost conflicted with a practical single-model deployment ([ADR 0015](01-adrs/0015-try-ensembled-model-prediction.md)).

## Why indirect supervision was replaced

Spatial weighting tried to suppress background errors indirectly, but it did not tell the model clearly what constituted foreground. Adding an attention head created the missing output, and BCE created the first direct mask target. BCE was then replaced with focal loss because the mask was dominated by easy background pixels and the difficult false positives needed more gradient ([ADR 0016](01-adrs/0016-add-attention-head-at-model-end.md), [ADR 0017](01-adrs/0017-adjust-loss-function-add-bce.md), [ADR 0019](01-adrs/0019-add-spatially-weighted-loss.md), [ADR 0020](01-adrs/0020-replace-bce-with-focal-loss.md)).

The attention head also needed actual negative examples. Shanghai Part A contained almost no non-crowd images, so 30–40 high-resolution easy and hard negatives were added before the normal split. This produced the largest improvement in false-positive behavior; later, 10 negatives were removed when the expanded negative ratio caused underestimation ([ADR 0023](01-adrs/0023-add-negative-samples-to-shanghai-part-a.md), [ADR 0027](01-adrs/0027-remove-10-of-the-manually-added-negative-samples.md)).

Hard threshold gating was removed because a static value such as `sigmoid > 0.88` made every new run depend on manual calibration and made mistakes catastrophic. Soft multiplication preserves confidence information, while the final convolution can recover from imperfect attention ([ADR 0024](01-adrs/0024-replace-hard-gating-and-manual-threshold-with-soft-gating-in-attention-head.md)).

## Why the loss was reshaped

MSE was too sensitive to large residuals in density maps scaled up to roughly 1000, so Huber loss was adopted. A lower delta reduced the influence of the quadratic region and made training steadier ([ADR 0021](01-adrs/0021-replace-mse-with-huber.md), [ADR 0032](01-adrs/0032-decrease-huber-delta.md)).

The first count penalty failed because aggregate count error alone does not specify where density belongs. The later count-level Huber experiment kept SSIM and focal loss alongside the count term, which stabilized MBE while retaining spatial and mask supervision ([ADR 0018](01-adrs/0018-add-count-penalty-loss.md), [ADR 0030](01-adrs/0030-try-using-count-loss-with-ssim-and-focal-loss.md)). Because count, SSIM, and focal losses live on very different numerical scales, their weights were separated into scale-unification and penalty factors. Dynamic weighting was deferred after preliminary adaptive schemes made convergence less predictable ([ADR 0022](01-adrs/0022-try-dynamic-loss-weightning.md), [ADR 0031](01-adrs/0031-put-more-effort-to-scale-losses-and-hp.md)).

## Why operational shortcuts were removed

Heavy augmentation, dataset preloading, and full-resolution validation were all considered as ways to improve robustness or speed. Heavy augmentation introduced cost and instability; preloading did not improve end-to-end throughput convincingly; and full-resolution validation exceeded practical memory limits for large images. The retained solutions are moderate augmentation, ordinary data loading, batching with careful padding, and fixed five-crop validation ([ADR 0009](01-adrs/0009-stop-trying-to-fix-heavy-augmentation.md), [ADR 0025](01-adrs/0025-add-batching-ability-to-validation-and-test-dataloaders.md), [ADR 0028](01-adrs/0028-add-five-crops-for-validation.md), [ADR 0029](01-adrs/0029-preload-data-into-memory-when-training.md)).

Finally, script and export logging were removed because graph formats tied serving too tightly to exact operators, input shapes, and dependency versions. The PyFunc wrapper keeps preprocessing, postprocessing, parameters, weights, and source code together, which better matches the project’s evolving architecture ([ADR 0012](01-adrs/0012-give-up-script-and-export-logging-formats-for-pyfunc.md)).
