# Success Metrics

The project evaluates model quality using crowd-counting metrics that consider both the total count and the spatial structure of the predicted density field.

Important metrics and signals include:

- MAE and RMSE for absolute count error,
- Positive NAE for normalized error without zero-denominator instability,
- MBE for systematic over- or underestimation,
- SSIM and density-map inspection for spatial structure,
- Dice and IOU when the attention mask is evaluated,
- validation stability and comparison against prior decisions.

Because historical code paths differ, it is important to compare runs with awareness of dataset and preprocessing changes as well as raw metric values.

The project does not define one universal historical winner in this documentation. The monitored metric is configuration-dependent, so a published result must include its dataset, split, preprocessing, loss weights, and parameter serialization.
