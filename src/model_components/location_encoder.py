import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rff

LAT_MIN, LAT_MAX = 48.77, 48.97
LON_MIN, LON_MAX = 2.22, 2.47

def normalize(L, LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LON_MIN, LON_MAX=LON_MAX):
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
    def __init__(self, LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LON_MIN, LON_MAX=LON_MAX):
        super().__init__()
        self.hidden_dim = 1024
        self.LAT_MIN = LAT_MIN
        self.LAT_MAX = LAT_MAX
        self.LON_MIN = LON_MIN
        self.LON_MAX = LON_MAX

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
        x = normalize(x, LAT_MIN=self.LAT_MIN, LAT_MAX=self.LAT_MAX, LON_MIN=self.LON_MIN, LON_MAX=self.LON_MAX)
        state = self.start_token.expand(x.shape[0], -1)
        for i, layer in enumerate(self.rff_layers):
            rff_feat = layer(x)
            state = self.blocks[i](state, rff_feat)
        x = self.final_proj(state)
        return F.normalize(x, p=2, dim=1)
