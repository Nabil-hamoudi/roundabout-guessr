import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationEncoder(nn.Module):
    def __init__(self):
        super(LocationEncoder, self).__init__()
        self.proj = nn.Linear(2, 512)  # pas rff encore

        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        x = torch.deg2rad(x)
        x = self.proj(x)
        x = torch.sin(x)
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, frozen_backbone=False):
        super().__init__()

        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 512) 
        )

    def forward(self, x):
        output = self.backbone.forward_features(x)
        features = output["x_norm_clstoken"]

        embeddings = self.proj(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

class MixedEncoder(nn.Module):
    def __init__(self):
        super(MixedEncoder, self).__init__()
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

    def forward(self, img, loc):
        img_embed = self.image_encoder(img)
        loc_embed = self.location_encoder(loc)
        img_embed = F.normalize(img_embed, p=2, dim=1)
        loc_embed = F.normalize(loc_embed, p=2, dim=1)
        return img_embed, loc_embed