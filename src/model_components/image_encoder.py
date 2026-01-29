import torch
import torch.nn as nn
import torch.nn.functional as F

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