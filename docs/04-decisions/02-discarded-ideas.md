# Discarded Ideas

This page records ideas that were considered but rejected, or that were not worth adopting as the current path.

These are valuable because they reduce repeated confusion and stop future experiments from revisiting the same dead ends without context.

## Rejected directions

- [Sliding-window patch prediction](01-adrs/0014-try-sliding-window-patching-prediction.md) removed global perspective, scale, and density context, and produced substantially worse validation results.
- [Model ensembling](01-adrs/0015-try-ensembled-model-prediction.md) increased memory use, inference latency, and serving complexity without being a sensible primary path for a deployable model.
- [Spatially weighted loss](01-adrs/0019-add-spatially-weighted-loss.md) was an indirect substitute for foreground supervision and had little or slightly negative impact on false positives.
- The first [count penalty formulation](01-adrs/0018-add-count-penalty-loss.md) distorted density maps and worsened NAE and MAE when aggregate count supervision displaced pixel-level guidance.
- [Preloading and pretransforming data](01-adrs/0029-preload-data-into-memory-when-training.md) did not produce a convincing end-to-end speedup and added memory and maintenance cost.

## Superseded experiments

- [Coordinate attention](01-adrs/0006-give-up-coord-att.md) added compute without measurable validation improvement.
- [HRNet](01-adrs/0007-give-up-hrnet.md) plateaued near the observed validation performance while consuming substantial GPU memory and training time.
- [Heavy augmentation](01-adrs/0009-stop-trying-to-fix-heavy-augmentation.md) created latency, artifacts, and occasional numerical instability rather than dependable generalization gains.
- [Script and export model logging](01-adrs/0012-give-up-script-and-export-logging-formats-for-pyfunc.md) was replaced by PyFunc after TorchScript and graph-export compatibility problems.
- [BCE attention supervision](01-adrs/0017-adjust-loss-function-add-bce.md) was useful as a first explicit mask signal, but class imbalance and easy-negative dominance left hard false positives unresolved.
- [Dynamic loss weighting](01-adrs/0022-try-dynamic-loss-weightning.md) remains deferred because preliminary learnable and adaptive schemes made convergence less predictable.

## Not permanently closed

JHU-Crowd++ tuning is deferred rather than discarded. Likewise, count-loss research may be revisited, but only with spatial and structural losses retained. These distinctions matter: an experiment can be unsuitable for the current path without proving that the underlying idea can never work.
