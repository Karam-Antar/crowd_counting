import torch
from src.core.params import BaseParams
from src.models.hrnet.backbone import HRNetBackbone
from src.models.hrnet.net import HRNet
from src.models.submodules.coord_att import CoordAtt
from src.models.hrnet.neck import HRNetNeck
import torch.nn.functional as F

from src.models.encoder_decoder import EncoderDecoder


class CrowdCounter(torch.nn.Module):
    def __init__(self, params: BaseParams):
        super().__init__()
        
        # 1. Initialize Backbone
        if params.backbone.startswith('hrnet'):
            self.net = HRNet(params)
        else:
            self.net = EncoderDecoder(params)

    def forward(self, x):
        density_map = self.net(x)
        # Ensure no negative values in the density map
        density_map = F.softplus(density_map)
        
        return density_map