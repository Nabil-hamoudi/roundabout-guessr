import torch
import torch.nn as nn
import torch.nn.functional as F
import rff
import copy


#Projection prise du code de GeoCLIP !
#Pas nécessaire pour la France, on a quand même gardé
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

#https://amaarora.github.io/posts/2020-08-30-gempool.html 
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        #[Batch, N_patches, Channels] -> [Batch, Channels]
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = x.mean(dim=1)
        x = x.pow(1.0 / self.p)

        return x
    
class ResBlock(nn.Module):
    def __init__(self, hidden_dim, encoded_size=256, dropout=0.1):
        super().__init__()
        self.rff_proj = nn.Linear(encoded_size * 2, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim) 
        )
    def forward(self, current_state, rff_feat):
        freq_emb = self.rff_proj(rff_feat)
        x = current_state + freq_emb
        out = self.net(x)
        return x + out
    

class LocationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.sigmas = [2.0, 16.0, 64.0, 256.0]
        
        encoded_size = 256
        hidden_dim = 1024 
        
        self.rff_layers = nn.ModuleList([
            rff.layers.GaussianEncoding(sigma=s, input_size=2, encoded_size=encoded_size)
            for s in self.sigmas
        ])
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, encoded_size, dropout=0.1)
            for _ in self.sigmas
        ])
        
        self.start_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.final_proj = nn.Linear(hidden_dim, 512)

    def forward(self, x):
        x = equal_earth_projection(x)
        state = self.start_token.expand(x.shape[0], -1)
        for i, layer in enumerate(self.rff_layers):
            rff_feat = layer(x)
            state = self.blocks[i](state, rff_feat)
        x = self.final_proj(state)
        return F.normalize(x, p=2, dim=1)

class FeatureAdapter(nn.Module):
    #prends les features et adapte avec skip ! (pour translater un peu)
    #https://arxiv.org/pdf/1902.00751 nous dit qu'on pourrait même en mettre DANS dino
    #mais on a clairement pas assez de data un petit adapter à la fin c'est déjà pas mal
    def __init__(self, in_dim=384, bottleneck=96):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, in_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # Residual: x + scale * adapter(x)
        adapted = self.up(self.act(self.down(x)))
        return x + self.scale * adapted


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        #on gèle dino pour économiser de la VRAM
        #Et que finetune dino avec si peu de données c'est un peu dangereux
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.backbone.parameters():
            param.requires_grad = False 
        
        self.patch_adapter = FeatureAdapter(in_dim=384, bottleneck=96)
        self.cls_adapter = FeatureAdapter(in_dim=384, bottleneck=96)
        
        self.pool = GeM()
        
        #on fait une projection patch + cls -> embed dim
        #c'est lui et les adapters qui font tout le travail
        #le rendre plus profond serait dangereux, sauf avec resblocks
        self.proj = nn.Sequential(
            nn.Linear(768, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )
    
    def forward(self, x):
        with torch.no_grad():
            output = self.backbone.forward_features(x)
        
        patch_tokens = self.patch_adapter(output["x_norm_patchtokens"])
        cls_token = self.cls_adapter(output["x_norm_clstoken"])
        
        pooled_patches = self.pool(patch_tokens)
        combined = torch.cat([pooled_patches, cls_token], dim=1)
        
        embeddings = self.proj(combined)
        return F.normalize(embeddings, p=2, dim=1)

#il mixe les deux
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