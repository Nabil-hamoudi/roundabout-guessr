import torch
import torch.nn as nn
import torch.nn.functional as F
import rff
import copy

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180


class LocationEncoder(nn.Module):
    def __init__(self):
        super(LocationEncoder, self).__init__()

        sigmas = [1.0, 16.0,24.0, 256.0]
        embed_dim = 256*2
        
        self.rff_layers = nn.ModuleList([
            rff.layers.GaussianEncoding(sigma=s, input_size=2, encoded_size=256)
            for s in sigmas
        ])


        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in sigmas
        ])

        self.gates = nn.Sequential(
            nn.Linear(embed_dim * len(sigmas), len(sigmas)),
            nn.Softmax(dim=1)
        )

        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim*4, embed_dim*4),
            nn.LayerNorm(embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim*2),
            nn.LayerNorm(embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 512)
        )

    def forward(self, x):
        x = equal_earth_projection(x)
        #print(x)
        embeddings = [self.mlp[i](layer(x)) + layer(x)
                      for i, layer in enumerate(self.rff_layers)]
        x_s = torch.cat(embeddings, dim=1)
        #print(x_s.shape)

        x = self.final_proj(x_s)

        return F.normalize(x, p=2, dim=1)
    
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.backbone.parameters():
            param.requires_grad = False
        #for blk in self.backbone.blocks[-1:]:
        #    for param in blk.parameters():
        #        param.requires_grad = True
        
        self.proj = nn.Sequential(
            nn.Linear(384, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):
        output = self.backbone.forward_features(x)
        features = output["x_norm_clstoken"]
        #print(features.shape)

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