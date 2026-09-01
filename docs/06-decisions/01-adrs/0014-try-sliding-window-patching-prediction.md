# ADR 0013: Try sliding-window patch-based prediction

## Status

Rejected

## Context

Sliding-window patch-based inference is a common technique for handling large images in dense prediction tasks. The approach divides the image into overlapping patches, runs inference on each patch independently, and reconstructs the full prediction by merging patch outputs (e.g., averaging overlaps).

This technique was explored to:
- Reduce peak memory usage during inference
- Enable processing of arbitrarily large images
- Potentially improve localization by focusing on local details

## Decision

Abandon patch-based prediction. The approach degraded model performance severely because it eliminated the global spatial context essential for density estimation and scale reasoning.

## Consequences

- **Loss of spatial context**: Each patch is predicted in isolation, removing the model's ability to reason about global scene structure.
- **Poor scale reasoning**: Crowd density is perspective-dependent; people near the camera appear larger than those far away. Predicting patches independently prevents the model from learning this perspective relationship across the image.
- **Inconsistent distribution estimates**: The model relies on understanding the overall crowd distribution and density gradient across the scene. Patches destroy this signal.
- **Severe performance degradation**: Validation metrics (NAE, MAE) were significantly worse than full-image inference.

## Rationale

Crowd density estimation is fundamentally a **global prediction task**:

1. **Perspective and scale**: The relationship between pixel density and actual crowd count varies across the image due to camera perspective. A person near the bottom of the image occupies more pixels than a person at the top. The model learns this perspective relationship by seeing the entire image.

2. **Density gradients**: Crowds exhibit spatial structure—high density in some regions, sparse in others. Understanding these gradients requires seeing the full scene. Isolated patches cannot reason about whether they are in a "crowded zone" or a "sparse zone" relative to the overall scene.

3. **Boundary effects**: Dividing an image into patches introduces artificial boundaries. Crowds at patch boundaries are incorrectly processed, and the lack of surrounding context leads to poor density predictions near edges.

4. **Loss of learned spatial priors**: The encoder-decoder architecture learns rich spatial priors during training on full images. These priors depend on global context and cannot be effectively applied to small patches.

## Why Full-Image Inference is Necessary

Unlike semantic segmentation or object detection (which can operate locally), density estimation requires understanding:
- The scale of objects relative to the global context
- The spatial distribution and clustering of crowds
- The perspective transformation across the entire scene

For very large images that exceed GPU memory, the solution is not patch-based inference but:
- Resizing the image (with corresponding label adjustment)
- Using a more memory-efficient model variant
- Accumulating gradients across multiple resolution levels
- Increasing GPU memory or using multi-GPU strategies

Patch-based prediction sacrifices accuracy for convenience and is incompatible with the nature of the task.

## Related Decisions

- **ADR 0001** (Density-map regression): Density estimation is a global spatial task; patch-based approaches contradict this design.
- **ADR 0002** (Geometry-adaptive targets): Scale reasoning depends on full-image context; patches eliminate this signal.
- **ADR 0008** (Encoder-decoder architecture): The skip connections and feature pyramid of the encoder-decoder rely on multi-scale, full-image reasoning.
