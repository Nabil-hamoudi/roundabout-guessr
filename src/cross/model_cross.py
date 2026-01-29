import torch
import torch.nn as nn
import torch.nn.functional as F
import rff
import numpy as np
from src.model_components.image_encoder import ImageEncoder
from src.model_components.location_encoder import LocationEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAT_MIN, LAT_MAX = 48.77, 48.97
LON_MIN, LON_MAX = 2.22, 2.47

#il mixe les deux
class CrossEncoder(nn.Module):
    def __init__(self, LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LON_MIN, LON_MAX=LON_MAX):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder(LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LON_MIN, LON_MAX=LON_MAX)
        self.sat_encoder = ImageEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, img, loc, sat):
        img_embed = self.image_encoder(img)
        loc_embed = self.location_encoder(loc)
        sat_embed = self.sat_encoder(sat)
        return img_embed, loc_embed, sat_embed, self.logit_scale.exp()