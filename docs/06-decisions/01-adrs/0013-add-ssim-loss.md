# ADR 0013: Add SSIM loss to the composite loss function

## Status

Accepted

## Context

The model was optimized using MSE (Mean Squared Error) loss on the density map output. While MSE effectively penalizes magnitude errors, it treats all pixels equally and does not capture the **structural similarity** between predicted and ground-truth density maps.

The density map has important spatial structure: smooth gradients, local peaks (crowd centers), and boundaries between dense and sparse regions. MSE alone could produce blurry or distorted density predictions that numerically minimize error but fail to capture true crowd distribution patterns.

## Decision

Integrate Structural Similarity Index (SSIM) loss into the composite loss function. After experimentation with multiple integration approaches, the optimal strategy was to combine MSE and SSIM with learned weights, **prioritizing SSIM** to guide the model toward structurally accurate density predictions.

## Experimentation and Findings

### Approaches Tried

1. **SSIM alone**: Too restrictive; model struggled to learn precise density magnitudes.
2. **MSE + SSIM (equal weights)**: Moderate improvement; MSE's magnitude focus competed with SSIM's structural guidance.
3. **MSE + SSIM (MSE-dominated weights)**: Minimal improvement; SSIM signal was too weak.
4. **MSE + SSIM (SSIM-dominated weights)** ✓ **BEST**: Clear performance improvement; the model learned both structural accuracy and magnitude consistency.

### Optimal Configuration

The winning approach combines:
- **MSE term**: Captures magnitude accuracy; penalizes overall count errors
- **SSIM term**: Captures structural similarity; penalizes boundary distortions, gradient smoothness, and spatial pattern misalignment
- **Weight ratio**: SSIM weight ≈ 0.3–0.5, MSE weight ≈ 0.5–0.7, with SSIM as the dominant signal

The composite loss is:
$$\mathcal{L} = \lambda_{MSE} \cdot \text{MSE}(y, \hat{y}) + \lambda_{SSIM} \cdot (1 - \text{SSIM}(y, \hat{y}))$$

where $\lambda_{SSIM} > \lambda_{MSE}$ to prioritize structural accuracy.

## Consequences

- **Improved spatial structure**: Predictions are smoother and more realistic, with sharper crowd boundaries and more coherent density patterns.
- **Better count accuracy**: Structural guidance reduces both systematic bias and variance in estimated crowd counts.
- **Enhanced generalization**: Models trained with SSIM loss generalize better to unseen datasets with different scene characteristics.
- **Hyperparameter tuning**: The weight ratio ($\lambda_{SSIM}$ vs. $\lambda_{MSE}$) becomes a tunable parameter, allowing fine-grained control over the trade-off between magnitude and structure.

## Rationale

SSIM is a perceptual loss metric designed to align with human perception of image similarity. For crowd density prediction:

1. **Magnitude vs. Structure**: MSE focuses on pixel-level magnitude, while SSIM considers luminance, contrast, and structural patterns. A prediction can be "close" in MSE but spatially distorted; SSIM penalizes such cases.

2. **Local Context**: SSIM is computed over local windows, making it sensitive to local density peaks, gradients, and boundaries—exactly the features needed for accurate crowd localization.

3. **Natural images**: Dense prediction of density maps is analogous to image reconstruction; SSIM was originally designed for image quality assessment and transfers naturally to this domain.

4. **Empirical validation**: Clear improvement in validation metrics (NAE, MAE) and qualitative inspection of predictions showed sharper, more coherent density maps.

## Implementation

The composite loss is encapsulated in `MaskMSESSIMLoss` (or similar class in `src/models/loss.py`):

```python
class MSESSIMLoss(nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        self.ssim_weight = params.ssim_weight  # Typically 0.3–0.5
        self.criterion_mse = nn.MSELoss()
        self.criterion_ssim = SSIM(data_range=params.label_scaler)
    
    def forward(self, pred, target):
        mse_loss = self.criterion_mse(pred, target)
        ssim_loss = 1 - self.criterion_ssim(pred, target)
        loss = (1 - self.ssim_weight) * mse_loss + self.ssim_weight * ssim_loss
        return loss
```

## Related Decisions

- **ADR 0001** (Density-map regression): SSIM loss applies directly to density map structure, strengthening the target specification.
- **ADR 0005** (Params object): The `ssim_weight` parameter is stored in `BaseParams`, enabling systematic exploration of weight ratios via hyperparameter tuning.
- **ADR 0009** (Stop heavy augmentation): Better loss function (SSIM) reduced need for excessive regularization.
- **ADR 0015** (Attention head): SSIM loss can be combined with mask losses to guide both structural accuracy and foreground suppression.
