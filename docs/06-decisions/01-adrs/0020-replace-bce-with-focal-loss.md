# ADR 0020: Replace BCE with focal loss for attention head supervision

## Status

Accepted

## Context

The BCE loss (ADR 0017) provided explicit supervision for the attention head, but suffered from severe class imbalance: most pixels in the image are background (label=0), with only a small fraction being crowd (label=1).

BCE treats both classes equally, resulting in:
- The loss being dominated by easy background pixels (where the model trivially predicts 0)
- Hard negatives (false positives in near-crowd regions) receiving minimal gradient signal
- The model learning to suppress obvious background patterns but failing at the hard cases—precisely where false positives were most problematic

The model still struggled to isolate true crowd regions from textured backgrounds, shadows, and edge artifacts.

## Decision

Replace BCE loss with **Focal Loss**, which is designed specifically for imbalanced classification:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

where:
- $p_t$ is the predicted probability for the ground-truth class
- $\gamma$ (focus parameter) controls how much to down-weight easy examples
- $\alpha_t$ (balance parameter) can further adjust class weights

**Effect**: Focal loss down-weights easy examples (trivial background) and focuses training on hard negatives (false positives at boundaries and in textured regions).

## Result: Breakthrough in False Positive Suppression

Focal loss was the turning point: **the model finally began to isolate non-crowd patterns effectively**. 

Key improvements:
- False positives in textured backgrounds decreased significantly
- The attention head learned sharp, coherent foreground masks
- Crowd-background boundaries became crisper
- Validation metrics (NAE, MAE) improved noticeably

Unlike BCE, which spread gradient equally across all pixels, focal loss concentrated learning effort on the hard cases—exactly where the model needed guidance.

## Why Focal Loss Solved the Problem

1. **Addresses class imbalance**: Most pixels are background; focal loss prevents the easy negatives from dominating the gradient.

2. **Focuses on hard negatives**: The $(1 - p_t)^{\gamma}$ term down-weights confident predictions and amplifies uncertain or wrong predictions. This forces the model to improve precisely where it was failing (false positives in hard-to-classify regions).

3. **Separates signal from noise**: By de-emphasizing trivial background pixels (which the model already predicts well), focal loss allows the model to concentrate on learning the boundary between crowds and confusing textures.

4. **Implicit bootstrapping**: As the model improves on hard negatives, those examples become easier, focal loss automatically reduces their weight, and the model moves to even harder cases. This self-directed curriculum accelerates learning.

## Implementation

The focal loss replaces BCE in the composite loss:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # pred: sigmoid output [0, 1]
        # target: binary mask [0, 1]
        bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce
        return loss.mean()

class MaskMSESSIMLoss(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.ssim_weight = params.ssim_weight
        self.mask_loss_weight = params.mask_loss_weight
        self.criterion_mse = nn.MSELoss()
        self.criterion_ssim = SSIM(data_range=params.label_scaler)
        self.criterion_focal = FocalLoss(
            alpha=params.mask_loss_alpha,  # Typically 0.6–0.95
            gamma=params.mask_loss_gamma   # Typically 2.0–4.0
        )
    
    def forward(self, pred_density, pred_mask, target_density):
        # Density loss (MSE + SSIM)
        mse_loss = self.criterion_mse(pred_density, target_density)
        ssim_loss = 1 - self.criterion_ssim(pred_density, target_density)
        density_loss = (1 - self.ssim_weight) * mse_loss + self.ssim_weight * ssim_loss
        
        # Ground-truth mask
        target_mask = (target_density > 0).float()
        
        # Mask loss (Focal instead of BCE)
        mask_loss = self.criterion_focal(pred_mask, target_mask)
        
        # Composite loss
        loss = density_loss + self.mask_loss_weight * mask_loss
        return loss
```

## Hyperparameter Tuning

Focal loss introduces two hyperparameters now in `BaseParams`:
- **`mask_loss_alpha`** (typically 0.6–0.95): Weight for class imbalance; higher values emphasize positive class (crowds).
- **`mask_loss_gamma`** (typically 2.0–4.0): Focus parameter; higher values more aggressively down-weight easy examples.

Tuning these parameters enabled fine-grained control over how much the model focused on hard negatives.

## Significance

This decision marked the **transition from architectural experimentation (attention head) to loss function engineering success**. Unlike:
- **Coordinate attention (ADR 0006)**: Failed at architectural level
- **BCE loss (ADR 0017)**: Failed at loss level due to imbalance
- **Spatial weighting (ADR 0019)**: Failed at implicit supervision

Focal loss succeeded because it **directly addressed the root cause**: the model couldn't learn from imbalanced data. By re-weighting the loss to focus on hard negatives, the model finally developed the ability to distinguish subtle non-crowd patterns from actual crowds.

## Related Decisions

- **ADR 0015** (Add attention head): Architectural foundation; focal loss provides the learning signal it needed.
- **ADR 0017** (Add BCE loss): First attempt at mask supervision; focal loss improved it by handling imbalance.
- **ADR 0013** (Add SSIM loss): Structural guidance for density; focal loss provides similar precision for foreground suppression.
- **ADR 0005** (Params object): `mask_loss_alpha` and `mask_loss_gamma` parameters enable systematic tuning of focal loss.
