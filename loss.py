import torch.nn as nn
import torch.nn.functional as F
import torch
#https://discuss.pytorch.org/t/how-to-implement-a-custom-loss-in-pytorch/197938/3
#complètement piqué d'ici
class GeoTripletLoss(nn.Module):
    def __init__(self, alpha = 0.5, margin = 1):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
    
    def forward(self, img_a, img_b, img_c, pos_a, pos_c):

        d_ab = F.pairwise_distance(img_a, img_b)
        d_ac = F.pairwise_distance(img_a, img_c)

        d_coord = F.pairwise_distance(pos_a, pos_c)

        #On ajoute la distance dans la margin !
        dyn_margin = self.margin*(1 - torch.exp(-self.alpha*d_coord))

        #relu = max (v,0) ici
        loss = (F.relu(d_ab - d_ac + dyn_margin)).mean()

        return loss