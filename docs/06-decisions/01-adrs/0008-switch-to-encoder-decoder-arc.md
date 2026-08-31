# ADR 0008: Switch to encoder-decoder architecture

## Status

Accepted

## Context

Early crowd-counting approaches used custom backbone-head combinations or simpler feature extractors that struggled to capture sufficient spatial and semantic information for accurate density map regression.

## Decision

Adopt a standard encoder-decoder architecture (U-Net, MAnet, UnetPlusPlus) from `segmentation_models_pytorch` as the core feature extractor. The encoder is a pretrained backbone (e.g., EfficientNet, ConvNeXt) and the decoder progressively upsamples features, maintaining spatial detail through skip connections.

## Consequences

- **Immediate performance improvement**: The switch resulted in a substantial jump in validation metrics (e.g., val_nae dropped significantly) compared to prior architectures.
- **Became first-line baseline**: The encoder-decoder framework became the default architecture choice for all subsequent experiments and hyperparameter tuning.
- **Faster feature development**: Skip connections and multi-scale decoding eliminated the need for custom attention mechanisms or specialized spatial reasoning layers.
- **Backbone flexibility**: The modular design allows swapping backbones (EfficientNet, ConvNeXt, ConvNeXt-v2) without architectural refactoring.
- **Established baseline for comparison**: Future improvements are now measured against the encoder-decoder baseline rather than against custom alternatives.

## Rationale

Encoder-decoder architectures are the de facto standard for dense prediction tasks (segmentation, depth estimation, crowd density mapping). The pretrained encoders provide rich semantic features, and the decoder's skip connections preserve fine-grained spatial information essential for localizing crowds. The immediate performance gain validated this choice empirically and saved significant iteration time by establishing a strong, stable baseline.

This decision transformed the project from exploratory architectural search into focused hyperparameter tuning and loss function refinement.

## Related Decisions

- **ADR 0006** (Give up coordinate attention): Coordinate attention was an attempt to improve the earlier architecture; the encoder-decoder's skip connections made it redundant.
- **ADR 0007** (Give up HRNet): HRNet was an alternative backbone; the encoder-decoder with lighter backbones (EfficientNet, ResNet) proved more efficient.
- **ADR 0004** (Class inheritance): The `EncoderDecoder` module is instantiated within `CrowdCounter`, which is used by `BaseLitModel` as part of the inheritance-based training pipeline.
- **ADR 0005** (Params object): The `backbone`, `model_class`, `decoder_out_channels`, `decoder_attention_type`, and `trainable_backbone` parameters in `BaseParams` control encoder-decoder configuration.
