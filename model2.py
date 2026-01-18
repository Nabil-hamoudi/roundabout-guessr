import torch
import torch.nn as nn
import torch.nn.functional as F
import rff

class LocationEncoder(nn.Module):
    def __init__(self):
        super(LocationEncoder, self).__init__()

        sigmas = [1.0, 10.0, 100.0]
        
        self.rff_layers = nn.ModuleList([
            rff.layers.GaussianEncoding(sigma=s, input_size=2, encoded_size=256)
            for s in sigmas
        ])

        input_dim = len(sigmas) * 256 * 2 

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        x = torch.deg2rad(x)
        
        embeddings = [layer(x) for layer in self.rff_layers]
        
        x = torch.cat(embeddings, dim=1) 
        
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)
        return x
class ImageEncoder(nn.Module):
    def __init__(self):
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
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, img, loc):
        img_embed = self.image_encoder(img)
        loc_embed = self.location_encoder(loc)
        return img_embed, loc_embed, self.logit_scale.exp()