# ADR 0024: Replace hard gating with soft gating in attention head

## Status

Accepted

## Context

The attention head (ADR 0015) was originally implemented as a **hard gating mechanism**:

```python
# Original hard gating approach
attention_sigmoid = decoder_attention_output  # sigmoid output ∈ [0, 1]
mask_binary = (attention_sigmoid > threshold).float()  # Hard threshold, e.g., > 0.88
output = mask_binary * density_map  # Element-wise multiplication with binary mask
```

### Problems with Hard Gating

1. **Manual threshold tuning burden**: 
   - Had to statically choose threshold (e.g., 0.88) for each experiment run
   - No principled way to select optimal threshold
   - Different data/model combinations required different thresholds
   - Threshold changes forced re-running entire experiments

2. **Catastrophic failure mode**:
   - Binary threshold creates a sharp decision boundary
   - Any pixel on the wrong side of threshold (e.g., true foreground classified as background, or vice versa) creates a **zero contribution** (hard masking)
   - This destroys spatial structure: if true crowd pixels get masked out, the density prediction has no gradients from true positives
   - If background pixels slip through, false positives are unmasked and unrestricted
   - Result: Predictions could flip to opposite side of ground-truth

3. **Gradient flow issues**:
   - Binary threshold (step function) has zero gradient almost everywhere
   - Sigmoid has been flattened to 0 or 1, losing fine-grained confidence information
   - Model cannot learn gradual refinement of confidence scores

## Decision

Replace hard gating with **soft gating**:

```python
# New soft gating approach
attention_sigmoid = decoder_attention_output  # sigmoid output ∈ [0, 1]
gated_density = attention_sigmoid * density_map  # Element-wise multiplication with soft weights
output = conv_layer(gated_density)  # Final convolution on gated density
```

### Why Soft Gating Works Better

1. **No manual threshold needed**: 
   - Sigmoid values ∈ [0, 1] directly weight the density map
   - Confidence is continuous, not binary
   - No hyperparameter to tune per experiment

2. **Gradual degradation instead of catastrophic failure**:
   - If attention incorrectly assigns 0.5 confidence to a true crowd pixel: density gets attenuated by 50%, not zeroed out
   - If attention incorrectly assigns 0.8 confidence to a background pixel: density gets amplified by 80%, not unmuted
   - Errors are **proportional and reversible**, not binary and irreversible

3. **Final convolution provides error recovery**:
   - The final convolutional layer learns to post-process `(attention * density)`
   - If attention over-gates (suppresses true signal), conv can learn to amplify
   - If attention under-gates (releases false signal), conv can learn to suppress or smooth
   - Conv operates on the **full feature map** and can use spatial context to fix localized attention mistakes

4. **Smooth gradient flow**:
   - Soft multiplication preserves sigmoid gradients ∈ (0, 1)
   - Information about confidence uncertainty flows through to loss function
   - Model learns to calibrate attention confidence, not just make binary decisions

## Implementation Details

**Architecture change**:
```python
# Decoder output splits into two branches:
# 1. density_head: Conv layers → density_map (raw regression output)
# 2. attention_head: Conv layers → sigmoid(attention_sigmoid)

# Then:
gated = attention_sigmoid * density_map  # Soft element-wise gating
output = final_conv(gated)              # Learned post-processing
```

**Loss function interaction**:
- Focal loss still applies to attention head directly (to improve `attention_logits` calibration)
- Primary loss (Huber + SSIM) applies to final output after gating and conv
- Attention head learns to produce good soft masks; final conv learns to use them effectively

## Results

- Eliminated manual threshold tuning: No more per-experiment hyperparameter selection
- More robust predictions: False predictions don't catastrophically flip output
- Improved model stability: Smoother learning dynamics, better gradient flow
- Simplified model deployment: No threshold config needed at inference time

## Why This Matters

Hard gating represented a **binary thinking** approach: a pixel is either "crowd" or "not crowd", and the decision was absolute. 

Soft gating embraces **confidence-aware** reasoning: the attention head expresses *how confident* it is about each pixel, and the density map is attenuated proportionally. The final conv layer then makes the final decision, using spatial context and learned patterns to resolve ambiguous regions.

This mirrors how humans might approach crowd counting:
- Look at each pixel region: "I'm 70% sure this is a person-shaped feature"
- Estimate density there, but discount by confidence: `0.7 * density`
- Check the neighborhood: does the spatial pattern make sense for 70% confidence?
- If all neighbors agree, keep it; if isolated, down-weight it (conv learns this)

## Related Decisions

- **ADR 0015** (Add attention head): Original hard-gating mechanism; this ADR refines it.
- **ADR 0020** (Replace BCE with focal loss): Focal loss now operates on sigmoid confidence; soft gating preserves this confidence signal for loss computation.
- **ADR 0021** (Replace MSE with Huber loss): Huber loss applied after final conv; soft gating's gradual errors are better suited to Huber's robustness region.
- **ADR 0023** (Add negative samples): Negative samples give attention head diverse examples to calibrate soft confidence scores.

## Future Tuning Opportunities

While soft gating eliminates the threshold hyperparameter, potential future refinements:
- Learn the gating function (instead of linear 1×1 multiplication): could be learnable transformation
- Adaptive confidence weighting based on image statistics
- Multiple attention heads with different receptive fields, soft-gated and fused
