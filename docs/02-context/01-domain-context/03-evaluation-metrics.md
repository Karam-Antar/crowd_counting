# Evaluation Metrics

Evaluation needs to capture more than just the final count. A model may produce a reasonable count while generating a poor density field.

- **MAE** measures the average absolute difference between predicted and ground-truth counts.
- **RMSE** penalizes larger count errors more strongly.
- **Positive NAE** computes normalized absolute error only for non-zero ground-truth counts, avoiding division by zero or near-zero values.
- **MBE** preserves the sign of prediction error: positive values indicate overestimation and negative values indicate underestimation.
- **SSIM** evaluates structural similarity between predicted and target density maps.
- **Dice and IOU** can evaluate the foreground mask produced by the attention head.

Counts are obtained by summing the density map, with the training label scale accounted for. Metrics must be interpreted together with the dataset, split, preprocessing, and loss configuration.
