import torch.nn as nn
import torch.nn.functional as F
import torch

#from claude, merci claude !!!
def haversine_distance(pos1, pos2):
    lat1, lon1 = pos1[:, 0], pos1[:, 1]
    lat2, lon2 = pos2[:, 0], pos2[:, 1]
    
    R = 6371  # Rayon Terre en km
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dphi = torch.deg2rad(lat2 - lat1)
    dlambda = torch.deg2rad(lon2 - lon1)
    
    a = torch.sin(dphi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dlambda/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    
    return R * c


#https://discuss.pytorch.org/t/how-to-implement-a-custom-loss-in-pytorch/197938/3
#complètement piqué d'ici
class GeoTripletLoss(nn.Module):
    def __init__(self, alpha = 2, margin = 1):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
    
    def forward(self, img_a, img_b, img_c, pos_a, pos_c):

        d_ab = F.pairwise_distance(img_a, img_b)
        d_ac = F.pairwise_distance(img_a, img_c)

        d_coord = haversine_distance(pos_a, pos_c)/4500 #jsp on parcourt l'europe, y'aura p-ê des > 1 mais ça devrait aller

        #On ajoute la distance dans la margin !
        dyn_margin = self.margin*(1 - torch.exp(-self.alpha*d_coord))
        #print(f"Dist Pos (A-B): {d_ab.mean().item():.4f} | Dist Neg (A-C): {d_ac.mean().item():.4f} | Margin: {dyn_margin.mean().item():.4f}")
        #relu = max (v,0) ici
        loss = (F.relu(d_ab - d_ac + dyn_margin)).mean()

        return loss

class BatchHardLoss(nn.Module):
    def __init__(self, base_margin=1, geo_scale=1000.0, max_margin_add=2):
        super().__init__()
        self.base_margin = base_margin
        # geo_scale : distance en km à partir de laquelle la pénalité est max
        self.geo_scale = geo_scale 
        self.max_margin_add = max_margin_add

    def forward(self, anchors, positives, gps_positions):
        # anchors: [B, Dim]
        # gps_positions: [B, 2] (Lat, Long)

        # 1. Calcul de la distance visuelle positive (On veut -> 0)
        d_pos = F.pairwise_distance(anchors, positives)

        # 2. Matrice de distances visuelles de tout le batch
        # dist_matrix[i, j] = dist(anchor[i], anchor[j])
        dist_vis = torch.cdist(anchors, anchors, p=2)
        
        # Masquer la diagonale (distance à soi-même)
        eye = torch.eye(dist_vis.size(0)).to(dist_vis.device)
        dist_vis = dist_vis + eye * 1e6

        # 3. Trouver l'index du "Hard Negative" pour chaque image
        # C'est l'image j qui minimise la distance visuelle avec i
        d_neg, neg_indices = dist_vis.min(dim=1)

        # 4. Calculer la distance Géographique avec ce "Hard Negative" spécifique
        # On récupère la pos GPS de l'anchor[i] et de l'anchor[neg_indices[i]]
        pos_anchors = gps_positions
        pos_negs = gps_positions[neg_indices]
        
        # Calcul Haversine (version Tensor rapide)
        d_geo_km = haversine_distance(pos_anchors, pos_negs)

        # 5. Marge Dynamique
        # Si d_geo est grand, la marge augmente -> Le modèle doit pousser plus fort
        # Si d_geo est petit (voisin), la marge reste base_margin -> Cool, c'est normal
        
        # Formule : Marge = 0.5 + (1.0 * (1 - exp(-dist/1000)))
        # À 0km -> Marge = 0.5
        # À 1000km -> Marge ≈ 1.1
        # À 3000km -> Marge ≈ 1.4
        dynamic_margin = self.base_margin + self.max_margin_add * (1 - torch.exp(-d_geo_km / self.geo_scale))

        # 6. Loss Finale
        loss = F.relu(d_pos - d_neg + dynamic_margin)
        
        return loss.mean()