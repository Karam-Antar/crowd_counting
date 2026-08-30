# Inference Pipeline

The inference pipeline is designed to generate predictions for new inputs while preserving the assumptions used during training.

The key requirement is consistency: if the model was trained with a specific normalization, label scaling, or padding behavior, that behavior must be preserved or clearly documented when used at inference time.
