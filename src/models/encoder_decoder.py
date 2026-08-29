import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from src.core.params import BaseParams

class EncoderDecoder(nn.Module):
    """Wrap a segmentation-model encoder-decoder as the feature extractor for density estimation.

    This module instantiates a segmentation model from ``segmentation_models_pytorch``,
    optionally freezes the encoder backbone, and returns the final decoder feature map
    used by the crowd-counting head. The expected input is a normalized RGB image batch.

    Attributes:
        params (BaseParams): Experiment configuration controlling the selected model,
            encoder, and fine-tuning behavior.
        en_de: A segmentation-model backbone-decoder instance configured for the current
            crowd-counting task.
    """
    def __init__(self, params: BaseParams):
        """Initialize the encoder-decoder feature extractor.

        Args:
            params (BaseParams): Parameter bundle containing model class, backbone,
                pretrained weight source, decoder configuration, and fine-tuning rules.
        """
        super().__init__()
        self.params = params
        
        # 1. Initialize SMP U-Net
        # The output of this will now act as a feature extractor.
        # Ensure params.neck_out_channels > 1 (e.g., 16 or 32) in your config.
        smp_class = getattr(smp, params.model_class)  
        self.en_de = smp_class(
            encoder_name=params.backbone,       
            encoder_weights=params.backbone_weights,
            decoder_attention_type=params.decoder_attention_type,           
            in_channels=3,
            classes=params.decoder_out_channels, 
        )
        # 2. Handle Backbone Freezing safely
        if not params.trainable_backbone:
            for param in self.en_de.encoder.parameters():
                param.requires_grad = False
                
        elif params.unfrozen_blocks:
            for param in self.en_de.encoder.parameters():
                param.requires_grad = False
                
            for name, param in self.en_de.encoder.named_parameters():
                if any(block_name in name for block_name in params.unfrozen_blocks):
                    param.requires_grad = True

    def forward(self, x):
        """Run an RGB image batch through the configured encoder-decoder network.

        Args:
            x (torch.Tensor): Normalized input tensor with shape ``(B, 3, H, W)``.

        Returns:
            torch.Tensor: Decoder output feature map produced by the segmentation model.
        """
        # 1. Extract rich semantic features from SMP
        features = self.en_de(x)
        
        # Returning only the final tensor prevents breaking your existing training loop
        return features