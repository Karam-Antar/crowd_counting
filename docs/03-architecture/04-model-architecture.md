# Model Architecture

The primary model is a pretrained encoder-decoder selected from the project’s supported `segmentation_models_pytorch` architectures, including U-Net, MAnet, and UnetPlusPlus. Backbones such as EfficientNet and ConvNeXt can be selected through `BaseParams`. Decoder skip connections preserve spatial detail while the encoder supplies semantic and multi-scale features.

The model predicts a density map rather than a scalar count. A count is obtained by summing the map, preserving the spatial information needed for dense, overlapping, perspective-distorted scenes.

The false-positive mitigation path adds an attention head that predicts a sigmoid foreground-confidence map. The confidence values softly gate the density map, and a final convolution learns to recover or suppress imperfect local attention. Focal loss supervises the attention output; Huber, SSIM, and count-related terms supervise the density result. Binary hard gating and manual thresholds were removed because errors could otherwise destroy valid density signal.
