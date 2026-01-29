import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *
from model import *
from embed_database import *
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime
import numpy as np
import random
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

NBR_EPOCH = 150
BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 32
BATCH_COMBINED = 600
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

def validate_retrieval(model, clean_train_loader, val_loader):
    """
    A partir d'un modèle, d'un loader d'entraînement SANS AUGMENTATIONS et d'un loader du set de validation
    """
    model.eval()
    
    print("Construction de la Galerie...")
    gallery_feats = []
    gallery_locs = []
    
    with torch.no_grad():
        for imgs, locs, sat_imgs in tqdm(clean_train_loader):
            imgs = imgs.to(DEVICE)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                feats, _, _, _ = model(imgs, locs.to(DEVICE), sat_imgs.to(DEVICE))
            
            gallery_feats.append(feats.cpu())
            gallery_locs.append(locs)

    gallery_feats = torch.cat(gallery_feats, dim=0) # [N_train, 512]
    gallery_locs = torch.cat(gallery_locs, dim=0)   # [N_train, 2]
    
    # Normalisation pour la distance cosinus
    gallery_feats = F.normalize(gallery_feats, p=2, dim=1)

    print("Lancement des requêtes sur le validation set...")
    query_feats = []
    query_true_locs = []
    
    with torch.no_grad():
        for imgs, locs, sat_imgs in tqdm(val_loader):
            imgs = imgs.to(DEVICE)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                feats, _, _, _ = model(imgs, locs.to(DEVICE), sat_imgs.to(DEVICE))
            
            query_feats.append(feats.cpu())
            query_true_locs.append(locs)

    query_feats = torch.cat(query_feats, dim=0)
    query_feats = F.normalize(query_feats, p=2, dim=1)
    query_true_locs = torch.cat(query_true_locs, dim=0)

    similarity = query_feats @ gallery_feats.T 
    _, top1_indices = similarity.max(dim=1) 
    
    pred_locs = gallery_locs[top1_indices]
    
    dist_meters = physical_dist(query_true_locs, pred_locs) * 1000  # en mètres
    
    median_error = torch.median(dist_meters).item()

    print(f"\n--- Résultats Retrieval ---")
    print(f"Erreur Médiane : {median_error:.1f} mètres")

    print(f"R@50m : {(dist_meters < 50).float().mean()*100:.1f}%")
    print(f"R@200m: {(dist_meters < 200).float().mean()*100:.1f}%")
    print(f"R@500m: {(dist_meters < 500).float().mean()*100:.1f}%") 
    print(f"R@1km : {(dist_meters < 1000).float().mean()*100:.1f}%") 
    
    return median_error

def panorama_split(image_paths, positions, sat_paths, val_split_pct=0.1, want_clean = False):
    """
    Sampling des panoramas
    
    IMPORTANT : On groupe par position exacte (Lat, Lon) pour s'assurer 
    que les vues  d'un même panorama restent ensemble  la fuite
    de données.
    """
    
    min_len = min(len(positions), len(image_paths), len(sat_paths))
    positions_np = np.array(positions)[:min_len]
    paths_np = np.array(image_paths)[:min_len]
    sat_paths_np = np.array(sat_paths)[:min_len]
    
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
    train_sat_paths = sat_paths_np[train_indices].tolist()
    
    val_paths = paths_np[val_indices].tolist()
    val_pos = positions_np[val_indices].tolist()
    val_sat_paths = sat_paths_np[val_indices].tolist()
    
    print(f"Total : {n_locs}")
    print(f"Train samples : {len(train_paths)} images (sur {n_train_locs} lieux)")
    print(f"Val samples   : {len(val_paths)} images (sur {len(val_loc_keys)} lieux)")
    print(f"Ratio effectif : {len(val_paths)/min_len:.1%}")

    train_dataset = ImagesPosDataset(
        train_paths, train_pos, train_sat_paths, want_index=False, is_train=True
    )
    val_dataset = ImagesPosDataset(
        val_paths, val_pos, val_sat_paths, want_index=False, is_train=False
    )
    train_dataset_clean = ImagesPosDataset(
        train_paths, train_pos, train_sat_paths, want_index=False, is_train=False)

    if want_clean:
        return train_dataset, val_dataset, train_dataset_clean
    
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

    coef_img_loc = 0.4
    coef_img_sat = 0.4
    coef_sat_loc = 0.2

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
    train_dataset, val_dataset, train_dataset_clean = panorama_split(imgs, pos, sat_paths, want_clean=True)

    #val_dataset = copy.deepcopy(val_dataset)
    #val_dataset.dataset.is_train = False
    
    print(train_dataset.is_train)
    print(val_dataset.is_train)

    model = MixedEncoder().to(DEVICE)
    #model.load_state_dict(torch.load("model_epoch_3.pt"))

    #Gérer la validation après déjà je veux faire en sorte que ça forward
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    train_clean_loader = DataLoader(train_dataset_clean, batch_size=INFERENCE_BATCH_SIZE, num_workers=2)
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
            #break

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
        #loss = model_validation(model, val_loader)#, criterion)
        loss = model_validation(model, val_loader)#, criterion)

        if epoch % 5 == 0 and epoch > 0:
            loss = validate_retrieval(model, train_clean_loader, val_loader)
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