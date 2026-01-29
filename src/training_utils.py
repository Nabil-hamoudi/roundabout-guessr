from collections import defaultdict
import random
import torch
import numpy as np
from src.cross_view.dataset_cross import CrossDataset
from src.base.dataset import ImagesPosDataset


def cross_entropy_with_mask(logits, mask):
    masked_logits = logits.clone()
    masked_logits[~mask] = -float('inf')
    
    positive_logits = torch.diagonal(logits)
    
    log_sum_exp = torch.logsumexp(masked_logits, dim=1)
    loss = -(positive_logits - log_sum_exp)
    return loss.mean()

def compute_pair_loss(emb_a, emb_b, logit_scale, valid_mask):
    #[B, B]
    logits_ab = logit_scale * (emb_a @ emb_b.T)
    logits_ba = logits_ab.T
    
    loss_a = cross_entropy_with_mask(logits_ab, valid_mask)
    loss_b = cross_entropy_with_mask(logits_ba, valid_mask)
    return (loss_a + loss_b) / 2

def criterion_duplicates(img_embed, loc_embed, logit_scale, positions, sat_embed = None):
    B = img_embed.size(0)
    #print(positions.shape)
    pos_i = positions.unsqueeze(1)
    pos_j = positions.unsqueeze(0)

    # < 10m
    is_duplicate = (torch.abs(pos_i - pos_j) < 1e-4).all(dim=2)
    
    #on récup la diag et on peuple la matrice des masques
    eye = torch.eye(B, device=img_embed.device, dtype=torch.bool)
    valid_mask = eye | (~is_duplicate)

    #pas de sat = juste img-loc
    if sat_embed is None:
        loss = compute_pair_loss(img_embed, loc_embed, logit_scale, valid_mask)
        return loss
    
    coef_img_loc = 0.4
    coef_img_sat = 0.4
    coef_sat_loc = 0.2

    loss_img_loc = compute_pair_loss(img_embed, loc_embed, logit_scale, valid_mask)
    loss_img_sat = compute_pair_loss(img_embed, sat_embed, logit_scale, valid_mask)
    loss_sat_loc = compute_pair_loss(sat_embed, loc_embed, logit_scale, valid_mask) 

    total_loss = coef_img_loc * loss_img_loc + coef_img_sat * loss_img_sat + coef_sat_loc * loss_sat_loc
    
    return total_loss

def physical_dist(latlon1, latlon2):
    R = 6371.0
    dlat = torch.deg2rad(latlon1[:, 0] - latlon2[:, 0]) #[Batch, 2] pour tous les latlon
    dlon = torch.deg2rad(latlon1[:, 1] - latlon2[:, 1])
    a = torch.sin(dlat/2)**2 + torch.cos(torch.deg2rad(latlon1[:, 0])) * \
        torch.cos(torch.deg2rad(latlon2[:, 0])) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return R * c

def geographic_split(image_paths, positions, val_split_pct=0.1, cell_size_deg=0.02):
    """
    Args:
        cell_size_deg (float): 0.02 deg ~= 2.2 km. 
    """
    
    positions_np = np.array(positions)
    #print(len(positions))
    #print(len(image_paths))
    min_len = min(len(positions), len(image_paths))
    positions_np = positions_np[:min_len]
    paths_np = np.array(image_paths)[:min_len]
    
    lats = positions_np[:, 0]
    lons = positions_np[:, 1]
    
    grid_x = (lats // cell_size_deg).astype(int)
    grid_y = (lons // cell_size_deg).astype(int)
    
    #on convertit val_split_pck en vrai pct
    mod_factor = int(1 / val_split_pct)
    
    #pour motif en "diag"
    is_val_mask = ((grid_x + grid_y) % mod_factor) == 0
    
    train_paths = paths_np[~is_val_mask].tolist()
    train_pos = positions_np[~is_val_mask].tolist()
    
    val_paths = paths_np[is_val_mask].tolist()
    val_pos = positions_np[is_val_mask].tolist()
    
    print(f"\n--- Geographic Split (Cell: ~{cell_size_deg*111:.1f} km) ---")
    print(f"Total images   : {len(paths_np)}")
    print(f"Train samples  : {len(train_paths)}")
    print(f"Val samples    : {len(val_paths)} ({len(val_paths)/len(paths_np):.1%})")
    
    unique_train_cells = len(set(zip(grid_x[~is_val_mask], grid_y[~is_val_mask])))
    unique_val_cells = len(set(zip(grid_x[is_val_mask], grid_y[is_val_mask])))
    print(f"Cellules actives (Zones géo distinctes) : Train={unique_train_cells}, Val={unique_val_cells}")

    train_dataset = ImagesPosDataset(
        train_paths, train_pos, want_index=False, is_train=True
    )
    
    val_dataset = ImagesPosDataset(
        val_paths, val_pos, want_index=False, is_train=False
    )
    
    return train_dataset, val_dataset


def panorama_split(image_paths, positions, sat_paths = None, val_split_pct=0.1, want_clean = False):
    """
    Sampling des panoramas
    
    IMPORTANT : On groupe par position exacte (Lat, Lon) pour s'assurer 
    que les vues  d'un même panorama restent ensemble  la fuite
    de données.
    UNIQUEMENT COMPATIBLE SI SAT, CE NEST PAS UNE BLAGUE
    -> compatible avec les 2 finalement
    """
    
    min_len = min(len(positions), len(image_paths), len(sat_paths)) if sat_paths is not None else min(len(positions), len(image_paths))
    positions_np = np.array(positions)[:min_len]
    paths_np = np.array(image_paths)[:min_len]
    sat_paths_np = np.array(sat_paths)[:min_len] if sat_paths is not None else None
    
    print(f"\n--- Random Location Split ---")

    # Clé = (Lat, Lon) arrondi, Valeur = liste des indices
    location_groups = defaultdict(list)
    
    for idx, pos in enumerate(positions_np):
        loc_key = (round(pos[0], 6), round(pos[1], 6))
        location_groups[loc_key].append(idx)
        
    unique_locations = list(location_groups.keys())
    random.seed(42)
    random.shuffle(unique_locations)
    
    #Calcul du Split sur les LIEUX (et pas les images)
    n_locs = len(unique_locations)
    n_val_locs = int(n_locs * val_split_pct)
    n_train_locs = n_locs - n_val_locs
    
    train_loc_keys = unique_locations[:n_train_locs]
    val_loc_keys = unique_locations[n_train_locs:]
    
    train_indices = []
    for key in train_loc_keys:
        train_indices.extend(location_groups[key])
        
    val_indices = []
    for key in val_loc_keys:
        val_indices.extend(location_groups[key])
        
    train_paths = paths_np[train_indices].tolist()
    train_pos = positions_np[train_indices].tolist()
    train_sat_paths = sat_paths_np[train_indices].tolist() if sat_paths_np is not None else None
    
    val_paths = paths_np[val_indices].tolist()
    val_pos = positions_np[val_indices].tolist()
    val_sat_paths = sat_paths_np[val_indices].tolist() if sat_paths_np is not None else None
    
    print(f"Total : {n_locs}")
    print(f"Train samples : {len(train_paths)} images (sur {n_train_locs} lieux)")
    print(f"Val samples   : {len(val_paths)} images (sur {len(val_loc_keys)} lieux)")
    print(f"Ratio effectif : {len(val_paths)/min_len:.1%}")
    if sat_paths is not None:
        #print("On utilise les sat")
        train_dataset = CrossDataset(
            train_paths, train_pos, train_sat_paths, want_index=False, is_train=True
        )
        val_dataset = CrossDataset(
            val_paths, val_pos, val_sat_paths, want_index=False, is_train=False
        )
        train_dataset_clean = CrossDataset(
            train_paths, train_pos, train_sat_paths, want_index=False, is_train=False)
    else:
        train_dataset = ImagesPosDataset(
            train_paths, train_pos, want_index=False, is_train=True
        )
        val_dataset = ImagesPosDataset(
            val_paths, val_pos, want_index=False, is_train=False
        )
        train_dataset_clean = ImagesPosDataset(
            train_paths, train_pos, want_index=False, is_train=False)

    if want_clean:
        return train_dataset, val_dataset, train_dataset_clean
    
    return train_dataset, val_dataset