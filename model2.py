import torch
import torch.nn as nn
import torch.nn.functional as F
import rff

class LocationEncoder(nn.Module):
    def __init__(self):
        super(LocationEncoder, self).__init__()

        sigmas = [1.0, 16.0, 256.0]
        embed_dim = 256*2
        
        self.rff_layers = nn.ModuleList([
            rff.layers.GaussianEncoding(sigma=s, input_size=3, encoded_size=256)
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
            nn.Linear(embed_dim*3, embed_dim*2),
            nn.LayerNorm(embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 512)
        )

    def forward(self, x):
        
        lat, lon = torch.deg2rad(x[:, 0]), torch.deg2rad(x[:, 1])
        
        #en coo sphériques
        sx = torch.cos(lat) * torch.cos(lon)
        sy = torch.cos(lat) * torch.sin(lon)
        sz = torch.sin(lat)
        
        sphere_coords = torch.stack([sx, sy, sz], dim=1)
        embeddings = [self.mlp[i](layer(sphere_coords)) + layer(sphere_coords)
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
            nn.Linear(384, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512) 
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