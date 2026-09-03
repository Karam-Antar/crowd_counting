# ADR 0017: Adjust loss function to add BCE for attention head supervision

## Status

Accepted (experimental; foundation for future improvements)

## Context

Following the addition of an attention head (ADR 0015), the model needed explicit supervision to learn what regions contain crowds versus background. However, the attention head lacked a direct learning signal—it was only indirectly guided by the density map loss.

The core issue: the attention head outputs a sigmoid mask, but had no ground-truth mask target to compare against during training. Without supervision, the attention head could not reliably learn to suppress false positives.

## Decision

Restructure the model and loss function to create explicit supervision for the attention head:

1. **Model architecture**: Modify the model to output two tensors:
   - Density map (continuous, regression output)
   - Sigmoid attention mask (continuous, 0–1)

2. **Ground-truth mask generation**: Create a binary mask from the ground-truth density map:
   - Foreground: pixels where density > 0
   - Background: pixels where density = 0

3. **Composite loss function**: Apply two independent loss terms:
   - **MSE + SSIM loss** on the density map (as before)
   - **BCE (Binary Cross-Entropy) loss** on the attention head output vs. ground-truth mask

The combined loss is:
$$\mathcal{L} = \lambda_{density} \cdot \mathcal{L}_{MSE+SSIM}(\hat{y}_{density}, y_{density}) + \lambda_{mask} \cdot \text{BCE}(\hat{y}_{mask}, y_{mask})$$

where $y_{mask} = \mathbb{1}[y_{density} > 0]$ (indicator function for foreground).

## Consequences

- **Explicit supervision for attention**: The attention head now has a direct target (ground-truth mask) and gradient signal (BCE loss).
- **Foreground-background separation**: The model learns to discriminate crowds from background, reducing false positives.
- **Increased loss function complexity**: The loss now balances two competing objectives (density accuracy and foreground suppression).
- **Foundation for refinement**: BCE provides a starting point for more sophisticated mask loss strategies.

## Limitations & Findings

### Why BCE Was Not Optimal

1. **Mask granularity**: The binary mask (0 or 1) is too coarse. A pixel with density=0.01 and a pixel with density=10 are both "foreground," but the model should learn to predict very different confidence levels for each.

2. **Boundary artifacts**: BCE treats boundary pixels harshly. A pixel barely below the threshold (density=−0.01, masked as background) is penalized equally to a clearly empty region.

3. **Imbalanced targets**: Most pixels are background (0); BCE on imbalanced data requires class weighting or focal variants to train effectively.

4. **Independent losses**: The density loss and BCE loss are not aligned. The density loss might encourage a pixel's density toward 0.5 (uncertainty), while BCE wants the mask to be confident (0 or 1).

### Limited Experimentation

The team did not extensively tune or refine the BCE approach because:
- Initial results showed modest improvements without breakthrough gains
- The false positive problem remained partially unsolved
- Other loss function innovations (focal loss, calibrated mask loss, Huber regression) were recognized as more promising

## Path Forward

BCE was the **first step toward structured foreground-background learning**, but better approaches include:

1. **Focal Loss on mask**: Emphasize hard negatives (false positives in background) with focal weighting
2. **Calibrated mask loss**: Smooth the binary mask based on density magnitude (high density → high confidence, low density → lower confidence)
3. **Uncertainty-aware loss**: Learn confidence alongside density, allowing the model to express uncertainty at boundaries

## Implementation

The loss function integration in `src/models/loss.py`:

```python
class MaskMSESSIMLoss(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.ssim_weight = params.ssim_weight
        self.mask_loss_weight = params.mask_loss_weight
        self.criterion_mse = nn.MSELoss()
        self.criterion_ssim = SSIM(data_range=params.label_scaler)
        self.criterion_bce = nn.BCELoss()  # or BCEWithLogitsLoss
    
    def forward(self, pred_density, pred_mask, target_density):
        # Density loss (MSE + SSIM)
        mse_loss = self.criterion_mse(pred_density, target_density)
        ssim_loss = 1 - self.criterion_ssim(pred_density, target_density)
        density_loss = (1 - self.ssim_weight) * mse_loss + self.ssim_weight * ssim_loss
        
        # Ground-truth mask: foreground if density > 0
        target_mask = (target_density > 0).float()
        
        # Mask loss (BCE)
        mask_loss = self.criterion_bce(pred_mask, target_mask)
        
        # Composite loss
        loss = density_loss + self.mask_loss_weight * mask_loss
        return loss
```

## Related Decisions

- **ADR 0015** (Add attention head): Architectural addition; this ADR provides the learning signal.
- **ADR 0013** (Add SSIM loss): Structural loss for density map; similar principle applied to mask supervision.
- **ADR 0005** (Params object): `mask_loss_weight` parameter controls the balance between density and mask supervision.
- **Future ADR**: More sophisticated mask losses (focal, calibrated) will likely supersede BCE.
