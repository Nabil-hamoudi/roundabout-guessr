import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from dataset import *
from model_cross import *
from embed_database import *
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

NBR_EPOCH = 150
BATCH_SIZE = 32
BATCH_COMBINED = 256
DATAFOLDER = "gen_fr"
DATAJSON = DATAFOLDER + "/coordinates_paris.json"
DATAIMAGES = DATAFOLDER + "/data_paris"
DATASAT = DATAFOLDER + "/sat_paris"

def physical_dist(latlon1, latlon2):
    R = 6371.0
    dlat = torch.deg2rad(latlon1[:, 0] - latlon2[:, 0]) #[Batch, 2] pour tous les latlon
    dlon = torch.deg2rad(latlon1[:, 1] - latlon2[:, 1])
    a = torch.sin(dlat/2)**2 + torch.cos(torch.deg2rad(latlon1[:, 0])) * \
        torch.cos(torch.deg2rad(latlon2[:, 0])) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return R * c

def model_validation(model, val_loader, distance_thresholds_km=[0.1, 0.25, 1, 2]):
    """
    Effectue la validation du modèle, calcule les embeddings pour tout le
    jeu de validation, puis calcule la matrice de similarité et les métriques
    de rappel à différentes distances.

    Ce n'est pas une métrique représentative de la vraie performance, mais
    ça reste un bon indicateur ig.
    Le bon indicateur pour arrêter l'entraînement n'a pas été déterminé, c'est
    au feeling.
    """
    model.eval()
    all_img_embeds = []
    all_pos_embeds = []
    all_coords = []
    
    print("Construction de la banque de validation...")
    with torch.no_grad():
        for img, pos, sat in tqdm(val_loader, desc="Encoding Val", leave=False):
            img = img.to(DEVICE)
            pos = pos.to(DEVICE)
            sat = sat.to(DEVICE)
            
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                img_out, pos_out, sat_out, _ = model(img, pos, sat)
            all_img_embeds.append(img_out.cpu())
            all_pos_embeds.append(pos_out.cpu())
            all_coords.append(pos.cpu())
            
    all_img_embeds = torch.cat(all_img_embeds, dim=0)  # [N, D]
    all_pos_embeds = torch.cat(all_pos_embeds, dim=0)
    all_coords = torch.cat(all_coords, dim=0)  # [N, 2]
    
    #calcul de la matrice de similarité
    sim_matrix = all_img_embeds @ all_pos_embeds.T  # [N, N]
    
    n_samples = sim_matrix.shape[0]
    
    recall_at_dist = {d: 0 for d in distance_thresholds_km}
    all_top1_errors = []
    
    for i in range(n_samples):
        true_coord = all_coords[i].unsqueeze(0)
        
        scores = sim_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True)
        
        top1_idx = sorted_indices[0].item()
        pred_coord = all_coords[top1_idx].unsqueeze(0)
        
        top1_dist = physical_dist(pred_coord, true_coord).item()
        all_top1_errors.append(top1_dist)
        
        for threshold in distance_thresholds_km:
            if top1_dist <= threshold:
                recall_at_dist[threshold] += 1
    
    print(f"\n--- Validation Geolocation (N={n_samples}) ---")
    for threshold in distance_thresholds_km:
        acc = 100 * recall_at_dist[threshold] / n_samples
        print(f"Recall@{threshold}km: {acc:.2f}%")
    
    avg_error = sum(all_top1_errors) / len(all_top1_errors)
    median_error = torch.median(torch.tensor(all_top1_errors)).item()
    
    print(f"\nErreur Top-1:")
    print(f"  Moyenne: {avg_error:.1f} km")
    print(f"  Médiane: {median_error:.1f} km")
    
    return median_error  #on retourne la médiane aps la moyenne !

def geographic_split(image_paths, positions, sat_paths, val_split_pct=0.1, cell_size_deg=0.004):
    """
    S'occupe de faire le split en grille du dataset.
    L'objectif est d'éviter la contamination entre train et val.
    cell_size_deg : taille d'une cellule en degrés (0.004° ~ 444m)
    val_split_pct : pourcentage de données à mettre en validation (ex: 0.1 = 10%)
    Renvoie : train_dataset, val_dataset
    """
    
    positions_np = np.array(positions)
    #print(len(positions))
    #print(len(image_paths))
    min_len = min(len(positions), len(image_paths), len(sat_paths))
    positions_np = positions_np[:min_len]
    paths_np = np.array(image_paths[:min_len])
    sat_paths_np = np.array(sat_paths[:min_len])
    
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
    train_sat_paths = sat_paths_np[~is_val_mask].tolist()
    val_paths = paths_np[is_val_mask].tolist()
    val_pos = positions_np[is_val_mask].tolist()
    val_sat_paths = sat_paths_np[is_val_mask].tolist()
    
    print(f"\n--- Geographic Split (Cell: ~{cell_size_deg*111:.1f} km) ---")
    print(f"Total images   : {len(paths_np)}")
    print(f"Train samples  : {len(train_paths)}")
    print(f"Val samples    : {len(val_paths)} ({len(val_paths)/len(paths_np):.1%})")
    
    unique_train_cells = len(set(zip(grid_x[~is_val_mask], grid_y[~is_val_mask])))
    unique_val_cells = len(set(zip(grid_x[is_val_mask], grid_y[is_val_mask])))
    print(f"Cellules actives (Zones géo distinctes) : Train={unique_train_cells}, Val={unique_val_cells}")

    train_dataset = ImagesPosDataset(
        train_paths, train_pos, train_sat_paths, want_index=False, is_train=True
    )
    
    val_dataset = ImagesPosDataset(
        val_paths, val_pos, val_sat_paths, want_index=False, is_train=False
    )
    
    return train_dataset, val_dataset

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

def criterion_duplicates(img_embed, loc_embed,sat_embed, logit_scale, positions):
    B = img_embed.size(0)
    pos_i = positions.unsqueeze(1)
    pos_j = positions.unsqueeze(0)

    # < 10m
    is_duplicate = (torch.abs(pos_i - pos_j) < 1e-4).all(dim=2)
    
    #on récup la diag et on peuple la matrice des masques
    eye = torch.eye(B, device=img_embed.device, dtype=torch.bool)
    valid_mask = eye | (~is_duplicate)

    coef_img_loc = 1.0
    coef_img_sat = 1.0
    coef_sat_loc = 0.3

    loss_img_loc = compute_pair_loss(img_embed, loc_embed, logit_scale, valid_mask)
    loss_img_sat = compute_pair_loss(img_embed, sat_embed, logit_scale, valid_mask)
    loss_sat_loc = compute_pair_loss(sat_embed, loc_embed, logit_scale, valid_mask) 

    total_loss = coef_img_loc * loss_img_loc + coef_img_sat * loss_img_sat + coef_sat_loc * loss_sat_loc
    
    return total_loss

def train_clip(nbr_epoch=NBR_EPOCH, batch_size=BATCH_SIZE, batch_combined=BATCH_COMBINED, data_json=DATAJSON, data_images=DATAIMAGES):
    NBR_EPOCH = nbr_epoch
    BATCH_SIZE = batch_size
    BATCH_COMBINED = batch_combined

    scaler = torch.amp.GradScaler(DEVICE_STR)
    print("Chargement des images")
    imgs = get_images_paths(data_images)
    print(len(imgs))

    print("Chargement du JSON rond pts")
    pos = get_images_pos(data_json)
    log_history = []

    sat_paths = get_images_paths(DATASAT)
    print(len(sat_paths), DATASAT)

    #train_indices, val_indices = torch.utils.data.random_split(
    #    range(len(dataset)), 
    #    [0.95, 0.05], 
    #    generator=generator1
    #)
    train_dataset, val_dataset = geographic_split(imgs, pos, sat_paths)

    #val_dataset = copy.deepcopy(val_dataset)
    #val_dataset.dataset.is_train = False
    
    print(train_dataset.is_train)
    print(val_dataset.is_train)

    model = MixedEncoder().to(DEVICE)
    #model.load_state_dict(torch.load("model_epoch_3.pt"))

    #Gérer la validation après déjà je veux faire en sorte que ça forward
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    criterion = nn.CrossEntropyLoss()#BatchHardLoss()
    optimizer = torch.optim.AdamW([
        # Adapters inside ImageEncoder
        {'params': model.image_encoder.parameters(), 'lr': 1e-4}, 
        {'params': model.sat_encoder.parameters(), 'lr': 1e-4}, 
        {'params': model.location_encoder.parameters(), 'lr': 1e-4}, 
        {'params': [model.logit_scale], 'lr': 1e-3} 
    ], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-5)  # Descend sur 30 epochs
    #TODO : c'est bien redondant on devrait faire une fonction poru calcul le batch accum
    for epoch in range(1, NBR_EPOCH):
        print("Début de l'epoch " + str(epoch) + " sur " + str(NBR_EPOCH))

        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)

        accum_pred_img = []
        accum_pred_pos = []
        accum_pred_sat = []
        accum_scale = []
        accum_pos_coords = []
        batch_count = 0

        for img, pos, sat in pbar:
            img = img.to(DEVICE)
            pos = pos.to(DEVICE)
            sat = sat.to(DEVICE)
            B = img.size(0)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                pred_img, pred_pos, pred_sat, scale = model(img, pos, sat)
            accum_pred_img.append(pred_img)
            accum_pred_pos.append(pred_pos)
            accum_pred_sat.append(pred_sat)
            accum_pos_coords.append(pos)
            accum_scale.append(scale)
            batch_count += 1

            if batch_count == BATCH_COMBINED // BATCH_SIZE:
                big_pred_img = torch.cat(accum_pred_img, dim=0)
                big_pred_pos = torch.cat(accum_pred_pos, dim=0)
                big_pos_coords = torch.cat(accum_pos_coords, dim=0)
                big_pred_sat = torch.cat(accum_pred_sat, dim=0)
                # Pour le scale, on prend la moyenne
                big_scale = torch.stack(accum_scale).mean()


                loss = criterion_duplicates(
                    big_pred_img, 
                    big_pred_pos, 
                    big_pred_sat,
                    big_scale, 
                    big_pos_coords,
                )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                total_loss += loss.item()

                accum_pred_img = []
                accum_pred_pos = []
                accum_scale = []
                accum_pos_coords = []
                accum_pred_sat = []
                batch_count = 0

        if batch_count > 0:
            big_pred_img = torch.cat(accum_pred_img, dim=0)
            big_pred_pos = torch.cat(accum_pred_pos, dim=0)
            big_pos_coords = torch.cat(accum_pos_coords, dim=0)
            big_pred_sat = torch.cat(accum_pred_sat, dim=0)
            big_scale = torch.stack(accum_scale).mean()

            loss = criterion_duplicates(
                big_pred_img, 
                big_pred_pos, 
                big_pred_sat,
                big_scale, 
                big_pos_coords,
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()

        print(f"Fin epoch {epoch} loss tr moyenne {total_loss/max(1, (len(train_loader)/(BATCH_COMBINED / BATCH_SIZE)))}")
        #print(f"logit_scale = {model.logit_scale.exp().item():.2f}")

        print("Début de la validation : ")
        loss = model_validation(model, val_loader)#, criterion)

        log_entry = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)/(BATCH_COMBINED/BATCH_SIZE)),
            "lr": optimizer.param_groups[0]['lr'],
            "logit_scale": model.logit_scale.exp().item(),
            "median_error_km": loss,
            "timestamp": datetime.now().isoformat()
        }
        log_history.append(log_entry)
        with open("training_log.json", "w") as f:
            json.dump(log_history, f, indent=2)
        scheduler.step()
        print(f"Validation terminée, loss : {loss}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")


if __name__ == "__main__":
    train_clip()