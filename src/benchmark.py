import torch
import numpy as np
import cv2
import math
from tqdm import tqdm
from src.cross_view.model_cross import *
from src.base.model import *
from src.base.dataset import compat_transform, get_images_pos, get_images_paths

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COORDS_PATH = "yo/coordinates.json"
EMBEDS_COORDS_PATH = "coordinates_embeds.json"
MODEL_PATH = "model_epoch_13.pt" 
DB_PATH = "embeddings_db.pt"


def haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance en km entre deux points (lat, lon)"""
    R = 6371.0 

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_database(path):
    print(f"Chargement de la DB depuis {path}...")
    data = torch.load(path, weights_only=False)
    
    return data

def run_benchmark(coords_path, embeds_coords_path, db_path, model_path, is_cross=True):
    print("--- Démarrage du Benchmark ---")

    print("Chargement des listes images et positions...")
    pos_list = get_images_pos(coords_path) 
    embeds_pos_list = get_images_pos(embeds_coords_path)
    img_paths = get_images_paths()

    if len(pos_list) != len(img_paths):
        min_len = min(len(pos_list), len(img_paths))
        pos_list = pos_list[:min_len]
        img_paths = img_paths[:min_len]
    
    if is_cross:
        model = CrossEncoder().to(DEVICE)
    else:
        model = MixedEncoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.base_encoder
    model.eval()

    db = load_database(db_path)
    distances = []
    
    print(f"Benchmark sur {len(img_paths)} images...")
    for i in tqdm(range(len(img_paths))):

        true_pos = pos_list[i] 
        img_path = img_paths[i]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: continue

        t_img = compat_transform(image=img)["image"].to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            query_vec = model.image_encoder(t_img).detach().cpu().numpy()

        res = db.find_elem(query_vec, top_k=1)
        
        if not res:
            continue

        pred_id = res[0]["id"]

        pred_pos = embeds_pos_list[pred_id] # [Lat, Lon]

        d = haversine(true_pos[0], true_pos[1], pred_pos[0], pred_pos[1])

        distances.append(d)

    
    distances = np.array(distances) * 1000
    
    print("\n" + "="*30)
    print("RÉSULTATS DU BENCHMARK")
    print("="*30)
    print(f"Images testées    : {len(distances)}")
    print(f"Erreur Moyenne    : {np.mean(distances):.2f} m")
    print(f"Erreur Médiane    : {np.median(distances):.2f} m") 
    print("-" * 20)
    
    print(f"Précision @ 10km  : {np.mean(distances <= 10000) * 100:.2f}%")
    print(f"Précision @ 2km   : {np.mean(distances <= 2000) * 100:.2f}%")
    print(f"Précision @ 1km   : {np.mean(distances <= 1000) * 100:.2f}%")
    print(f"Précision @ 500m  : {np.mean(distances <= 500) * 100:.2f}%")
    print(f"Précision @ 200m  : {np.mean(distances <= 200) * 100:.2f}%")
    print(f"Précision @ 100m  : {np.mean(distances <= 100) * 100:.2f}%")
    print(f"Précision @ 25m   : {np.mean(distances <= 25) * 100:.2f}%")
    
    print("="*30)