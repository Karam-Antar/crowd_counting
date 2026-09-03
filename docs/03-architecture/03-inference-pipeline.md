# Inference Pipeline

Inference reconstructs the model from the serialized `BaseParams` and checkpoint, applies the same image normalization used during training, pads inputs when the architecture requires stride-compatible dimensions, and runs the model in evaluation mode.

The density output is converted to a crowd count by summing its spatial values and reversing the training label scale. When the attention head is present, its sigmoid values are multiplied directly with the density output and processed by the final convolution; inference does not apply a manually selected binary threshold.

The key requirement is consistency: normalization, label scaling, model architecture, padding, and postprocessing must come from the same training configuration. The MLflow PyFunc wrapper packages the parameter JSON, weights, and source code needed to reproduce this behavior.
