# ADR 0021: Replace MSE with Huber loss in density map regression

## Status

Accepted

## Context

The density map regression loss previously used MSE (Mean Squared Error):

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

MSE is sensitive to outliers because large errors are quadratically amplified. In crowd-counting scenarios, occasional large prediction errors (e.g., predicting 100 when ground truth is 10, or vice versa) can dominate the loss and distort gradient flow.

The density map has a large dynamic range: pixel values span from 0 to 1000 (due to label scaling factor of 1000). Large errors are naturally more frequent in high-density regions. MSE penalizes these quadratically, which can:
- Create unstable gradients (very large errors cause exploding gradients)
- Bias the model to reduce a few extreme errors at the expense of overall prediction quality
- Miss the typical, moderate errors that actually matter for count accuracy

## Decision

Replace MSE with **Huber loss**, which is robust to outliers:

$$\mathcal{L}_{Huber}(\delta) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{\delta}{2}) & \text{if } |y - \hat{y}| > \delta
\end{cases}$$

where $\delta$ is the **Huber delta parameter** that controls the transition between quadratic (small errors) and linear (large errors).

### Critical Tuning: Low Delta Value

The key insight was using a **low delta value** ($\delta = 5$) relative to the density map scale ($[0, 1000]$):

- **High delta** (e.g., $\delta = 50$): Most errors fall in the quadratic region; behaves similarly to MSE.
- **Low delta** (e.g., $\delta = 5$): Even moderate errors (e.g., error = 10) fall in the linear region; behaves like L1 loss for most practical predictions.

With $\delta = 5$, the model treats:
- Small errors (< 5): Quadratically penalized (precise optimization)
- Moderate-to-large errors (≥ 5): Linearly penalized (robust to outliers)

This forces the model to focus on getting predictions in the right ballpark rather than obsessing over extreme outliers.

## Result: Significant Improvement

Replacing MSE with Huber (δ=5) produced a **clear, measurable improvement** in validation metrics:
- NAE decreased noticeably
- MAE improved
- Prediction stability increased
- Training became more stable (fewer loss spikes)

The low delta was crucial; preliminary tests with higher delta values showed minimal or no improvement.

## Why Low Delta Works

1. **Outlier robustness**: The linear regime (error > δ) ignores the quadratic penalty, preventing occasional large errors from dominating the gradient.

2. **Appropriate for crowded regions**: In high-density areas (density = 500–1000), a prediction error of ±10 is common and acceptable. MSE would quadratically penalize this; Huber's linear regime treats it reasonably.

3. **Gradient stability**: With low delta, gradients remain bounded even for large errors. MSE can produce gradients of arbitrary magnitude; Huber's maximum gradient magnitude is $\delta$, making optimization more stable.

4. **Matches task requirements**: For crowd counting, we care about total count accuracy, not pixel-level precision. Huber loss encourages the model to be approximately correct across the density map rather than perfect at every pixel.

## Parameter Configuration

The Huber delta is now controlled by `huber_delta` in `BaseParams`:

- **Typical range**: 2–10 (relative to label scaling factor 1000)
- **Best found**: $\delta = 5$
- **Justification**: Balances between precise small errors and robust treatment of outliers

## Implementation

The Huber loss replaces MSE in the composite loss:

```python
class MaskMSESSIMLoss(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.ssim_weight = params.ssim_weight
        self.mask_loss_weight = params.mask_loss_weight
        self.huber_delta = params.huber_delta or 5.0
        
        self.criterion_huber = nn.HuberLoss(delta=self.huber_delta, reduction='mean')
        self.criterion_ssim = SSIM(data_range=params.label_scaler)
        self.criterion_focal = FocalLoss(
            alpha=params.mask_loss_alpha,
            gamma=params.mask_loss_gamma
        )
    
    def forward(self, pred_density, pred_mask, target_density):
        # Density loss: Huber + SSIM
        huber_loss = self.criterion_huber(pred_density, target_density)
        ssim_loss = 1 - self.criterion_ssim(pred_density, target_density)
        density_loss = (1 - self.ssim_weight) * huber_loss + self.ssim_weight * ssim_loss
        
        # Ground-truth mask
        target_mask = (target_density > 0).float()
        
        # Mask loss (Focal)
        mask_loss = self.criterion_focal(pred_mask, target_mask)
        
        # Composite loss
        loss = density_loss + self.mask_loss_weight * mask_loss
        return loss
```

## Significance

Huber loss with low delta was a **simple but high-impact change**:
- Easy to implement (one-line substitution in the loss function)
- No architectural changes required
- Immediate, measurable improvement
- Stabilized training and validation curves

This demonstrates the principle that **proper loss function design is often more impactful than architectural complexity**. Switching from MSE to Huber was more effective than many earlier architectural explorations (coordinate attention, HRNet, etc.).

## Lessons

- **Outlier robustness matters**: In tasks with large dynamic ranges, robust loss functions (Huber, Smooth L1, L1) often outperform MSE.
- **Hyperparameter tuning is critical**: Low delta (5 vs. 50) made the difference; the same loss function with different settings can fail or succeed.
- **Scale-aware design**: The delta was tuned relative to the density map scale [0, 1000]; a delta of 5 is aggressive for this range but worked well.

## Related Decisions

- **ADR 0013** (Add SSIM loss): Structural loss paired with Huber for magnitude; the combination is more powerful than either alone.
- **ADR 0020** (Replace BCE with focal loss): Simultaneous refinement of mask loss; together, Huber + focal loss solved both density accuracy and false positive problems.
- **ADR 0005** (Params object): `huber_delta` parameter enables systematic exploration of this hyperparameter across tuning trials.
- **ADR 0001** (Density-map regression): Huber loss strengthens the regression objective by handling the large dynamic range of scaled density maps.
