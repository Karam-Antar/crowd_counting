# Inference

Inference should use the same assumptions as the training path whenever possible. This includes normalization, input preprocessing, and the model’s expected output format.

The most important rule is consistency: if the training pipeline changed, the inference pipeline should be checked to ensure it still matches the same operating contract.
