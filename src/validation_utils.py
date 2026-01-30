
import torch
from tqdm import tqdm
from src.training_utils import physical_dist
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

def model_validation(model, val_loader, distance_thresholds_km=[0.1, 0.25, 1, 2, 10, 100], has_sat = False):
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
        #un petit mal pour un grand bien
        if not has_sat:
            for img, pos in tqdm(val_loader, desc="Encoding Val", leave=False):
                img = img.to(DEVICE)
                pos = pos.to(DEVICE)
                
                with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                    img_out, pos_out, _ = model(img, pos)
                all_img_embeds.append(img_out.cpu())
                all_pos_embeds.append(pos_out.cpu())
                all_coords.append(pos.cpu())
        else:   
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
    COMPATIBLE UNIQUEMENT SUR LES MODÈLES AVEC SATELLITE
    """
    model.eval()
    
    print("Construction de la Galerie...")
    gallery_feats = []
    gallery_locs = []
    
    with torch.no_grad():
        for imgs, locs, sat_imgs in tqdm(clean_train_loader):
            imgs = imgs.to(DEVICE)
            locs = locs.to(DEVICE)
            sat_imgs = sat_imgs.to(DEVICE)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                #_, pos_out, _, _ = model(imgs, locs, sat_imgs)
                pos_out = model.location_encoder(locs)
            
            gallery_feats.append(pos_out.cpu())
            gallery_locs.append(locs.cpu())

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
    print(f"R@100m : {(dist_meters < 100).float().mean()*100:.1f}%")
    print(f"R@200m: {(dist_meters < 200).float().mean()*100:.1f}%")
    print(f"R@500m: {(dist_meters < 500).float().mean()*100:.1f}%") 
    print(f"R@1km : {(dist_meters < 1000).float().mean()*100:.1f}%") 
    print(f"R@10km : {(dist_meters < 10000).float().mean()*100:.1f}%") 
    print(f"R@100km : {(dist_meters < 100000).float().mean()*100:.1f}%") 
    
    return median_error
