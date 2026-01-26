import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class BaseEmbed(nn.Module):
    """
    Docstring for BaseEmbed
    Modèle de base, pouvant servir de premier test pour l'entraînement
    Sera médiocre, pas très grave ?
    """
    def __init__(self):
        super().__init__()
        # On reçoit du (B,C,W,H)
        # Sûrement du (B,3,W,H)
        self.resnet = timm.create_model('resnet50', pretrained=True, features_only=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        #la FC pour proj
        #on a 2048 de dim sur la dernière layer des features de resnet50
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128)
        )

    def forward(self, x):
        #print(x.shape)
        #plus besoin on utilise TensorV2
        #x = x.permute(0, 3, 1, 2)
        #print(x.shape)
        with torch.no_grad():
            x = self.resnet(x)
            x = x[-1] #ça retourne toutes les features le gros fou
            #si on peut lui dire de mettre que la dernière je prends pck sinon en inférence c chiant ?

            #là on flatten et FC sur un vecteur de taille 128 ?
            #on peut faire mieux après là c'est pour faire la pipeline
            x = self.pool(x)
            x = x.flatten(1)

        x = self.projection(x)

        x = F.normalize(x, p=2, dim=1)
        #print(x.shape)
        return x
