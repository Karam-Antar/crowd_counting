# ADR 0018: Add count penalty loss

## Status

Rejected (incomplete experimentation; promising direction with caveats)

## Context

While the MSE + SSIM density loss achieved good spatial structure, the final evaluation metric is **total crowd count** (integrated density map). The model could potentially improve count accuracy by adding a direct loss term that penalizes count error:

$$\text{Count Loss} = |\text{predicted count} - \text{ground truth count}|$$

The hypothesis was that direct supervision on the count would force the model to optimize for total accuracy, complementing the spatial structure guided by the density loss.

## Decision

Add a count penalty loss term as an auxiliary objective:

$$\mathcal{L}_{total} = \lambda_{density} \cdot \mathcal{L}_{MSE+SSIM} + \lambda_{count} \cdot |\sum \hat{y}_{density} - \sum y_{density}|$$

## Initial Result: Failure

The count penalty loss **degraded model performance** in a single trial:
- Validation NAE and MAE got worse
- Density maps became distorted and less coherent
- Total count accuracy did not improve as hoped

## Why It Failed

The failure reveals a critical insight about loss function design in dense prediction:

1. **Loss of guidance signal**: The density loss (MSE + SSIM) implicitly guides the model to learn what constitutes a person by penalizing every pixel. This pixel-level supervision teaches the model to distinguish people from backgrounds, walls, and shadows.

2. **Count loss removes spatial hints**: When the model is only penalized for total count, it has no direct signal about where or how to localize crowds. The model could achieve the correct total count by:
   - Distributing density uniformly across the image
   - Over-predicting in some regions and under-predicting in others
   - Predicting density in background regions (false positives)

3. **Broken learning feedback**: The pixel-level density loss acts as a "hint" that facilitates learning. Removing or diminishing this hint (by prioritizing count loss) made the model's optimization landscape flatter and more ambiguous, paradoxically making count learning harder, not easier.

4. **Spatial structure ↔ Count accuracy trade-off**: Good density maps naturally lead to good counts. Poor density maps (even if they integrate to the correct total) are harder to learn from and generalize worse to new data.

## Lessons Learned

- **Task decomposition matters**: In multi-task scenarios, not all auxiliary objectives help. Penalizing only the aggregate (count) without supervising the components (spatial distribution) can backfire.
- **Implicit supervision is powerful**: The density loss provides implicit guidance for learning crowd appearance and location. Direct count loss lacks this structure.
- **Loss alignment is critical**: Adding loss terms requires ensuring they push the model in complementary, not competing, directions.

## Path Forward: Multi-Component Loss Fusion

Rather than abandoning count supervision, future work should explore **better ways to integrate count loss** with other objectives:

### Option 1: Balanced 3-Component Loss
$$\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{MSE+SSIM} + \lambda_2 \cdot \mathcal{L}_{BCE\_mask} + \lambda_3 \cdot \mathcal{L}_{count}$$

With careful tuning of $\lambda_1, \lambda_2, \lambda_3$ to balance density accuracy, foreground suppression, and count accuracy.

### Option 2: Hierarchical Loss
First optimize density and mask (Phases 1–N), then introduce count penalty at later stages once the spatial foundation is solid.

### Option 3: Calibrated Count Loss
Instead of raw count error, use a **differentiable count loss** that couples density structure:
$$\mathcal{L}_{count\_calibrated} = |\sum \hat{y} - \sum y| \cdot \text{SpatialConsistency}(\hat{y})$$

penalizing count error more heavily if the spatial structure is poor.

## Why We Did Not Continue Experiments

The team prioritized other improvements (SSIM refinement, attention head tuning, data cleaning) over extensive count loss experimentation because:
- One trial showed clear degradation; no signal of improvement
- The existing MSE + SSIM + BCE loss was already functional
- Compute budget was better spent on other directions

However, the direction is **not abandoned**—count loss deserves revisiting with better loss composition and tuning.

## Related Decisions

- **ADR 0013** (Add SSIM loss): Pixel-level supervision is key; count loss alone loses this guidance.
- **ADR 0017** (Add BCE for attention head): Multi-component losses require careful balancing; count loss must be added with this in mind.
- **ADR 0001** (Density-map regression): The target is the density map, not the count; supervising the count directly may conflict with the primary objective.
- **Future ADR**: Comprehensive multi-task loss engineering (density + count + foreground) will be needed for further gains.
