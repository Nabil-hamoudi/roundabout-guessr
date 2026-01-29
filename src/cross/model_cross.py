import torch
import torch.nn as nn
import torch.nn.functional as F
import rff
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAT_MIN, LAT_MAX = 48.77, 48.97
LON_MIN, LON_MAX = 2.22, 2.47

def normalize(L):
    """
    Normalise les coordonnées (Lat, Lon) pour qu'elles tiennent
    dans l'intervalle [-1, 1] en se focalisant sur les bornes définies.
    """
    
    latitude = L[:, 0]
    longitude = L[:, 1]
    
    y = 2 * (latitude - LAT_MIN) / (LAT_MAX - LAT_MIN) - 1
    x = 2 * (longitude - LON_MIN) / (LON_MAX - LON_MIN) - 1
    
    #calcul du ratio de déformation
    lat_mean_rad = np.radians((LAT_MIN + LAT_MAX) / 2)
    aspect_ratio = np.cos(lat_mean_rad)
    
    x = x * aspect_ratio
    
    return torch.stack((x, y), dim=1)

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
        self.hidden_dim = 1024

        #pour les bons signaux on utilise généralement
        #largeur_zone/(2sigma) pour donner une idée de précision en tous les cmb de mètres
        #sigma = 64 -> ~~1km de précision
        self.sigmas = [1, 2, 4, 8, 16, 32, 64, 128]
        
        encoded_size = 256
        hidden_dim = self.hidden_dim 
        
        self.rff_layers = nn.ModuleList([
            rff.layers.GaussianEncoding(sigma=s, input_size=2, encoded_size=encoded_size)
            for s in self.sigmas
        ])
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, encoded_size, dropout=0.5)
            for _ in self.sigmas
        ])
        
        self.start_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.final_proj = nn.Linear(hidden_dim, 512)

    def forward(self, x):
        x = normalize(x)
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
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        for param in self.backbone.parameters():
            param.requires_grad = False 
        
        self.patch_adapter = FeatureAdapter(in_dim=384, bottleneck=96)
        self.cls_adapter = FeatureAdapter(in_dim=384, bottleneck=96)
        
        self.pool = GeM()
        
        dino_output_dim = 384
        hidden_dim = 2048

        #on fait une projection patch + cls -> embed dim
        #c'est lui et les adapters qui font tout le travail
        #le rendre plus profond serait dangereux, sauf avec resblocks
        self.proj = nn.Sequential(
            nn.Linear(dino_output_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 512)
        )
    
    def forward(self, x):
        with torch.no_grad():
            output = self.backbone.forward_features(x)
        
        pooled_patches = self.pool(self.patch_adapter(output["x_norm_patchtokens"]))
        cls_token = self.cls_adapter(output["x_norm_clstoken"])
        
        combined = torch.cat([pooled_patches, cls_token], dim=1)
        
        embeddings = self.proj(combined)
        return F.normalize(embeddings, p=2, dim=1)

#il mixe les deux
class MixedEncoder(nn.Module):
    def __init__(self):
        super(MixedEncoder, self).__init__()
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()
        self.sat_encoder = ImageEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, img, loc, sat):
        img_embed = self.image_encoder(img)
        loc_embed = self.location_encoder(loc)
        sat_embed = self.sat_encoder(sat)
        return img_embed, loc_embed, sat_embed, self.logit_scale.exp()