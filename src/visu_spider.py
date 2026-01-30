import sys
import os
# Fix du chemin pour l'import src
sys.path.append(os.getcwd())

import folium
import json
import random
import math
from pathlib import Path
from tqdm import tqdm
import re

# --- IMPORTS ---
from src.embed_database import init_model, use_model, DEVICE

# --- CONFIGURATION ---
MAXPOINT = 8 # Réduit à 100 pour éviter de surcharger la carte visuellement
MODEL_PATH = "model_paris_50k.pt"
EMBED_PATH = "embeddings_db.pt"
JSON_PATH = "dataset/coordinates.json"
DATA_DIR = "dataset/data/"
OUTPUT_FILE = "carte_erreurs_paris.html"

## temp
CROSS = True  # Utiliser le modèle cross-view
FRANCE = False  # Modèle restreint à la France
LAT_MIN_F, LAT_MAX_F = 41.3, 51.1
LON_MIN_F, LON_MAX_F = -5.1, 9.6

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_error_color(dist):
    """Retourne une couleur selon la distance d'erreur en mètres."""
    if dist < 100: return 'blue'    # incroyable
    if dist < 500: return 'green'    # Excellent
    if dist < 1000: return 'orange'   # Quartier correct
    if dist < 2000: return 'red'   # Quartier correct
    return 'purple'                     # Perdu

def extract_int_id(key_str):
    numbers = re.findall(r'\d+', str(key_str))
    if numbers: return int(numbers[-1])
    return None

def run_error_map(r_model):
    print(f"--- Génération Carte des Erreurs ({MAXPOINT} points) ---")

    # 1. Chargement JSON
    print("ici test")
    print(JSON_PATH)
    if not os.path.exists(JSON_PATH):
        print(f"JSON introuvable")
        return
    with open(JSON_PATH, 'r') as f:
        coords_db = json.load(f)

    # 2. Indexation
    id_map = {extract_int_id(k): k for k in coords_db.keys() if extract_int_id(k) is not None}

    # 3. Sélection
    path_obj = Path(DATA_DIR)
    selected_keys = random.sample(list(coords_db.keys()), min(MAXPOINT, len(coords_db)))
    
    # Création de la carte
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=13, tiles='OpenStreetMap')
    
    # FeatureGroups pour pouvoir filtrer sur la carte finale
    fg_lines = folium.FeatureGroup(name="Lignes d'erreur").add_to(m)
    fg_points = folium.FeatureGroup(name="Points de prédiction").add_to(m)

    # 4. Boucle de calcul
    for query_key in tqdm(selected_keys):
        try:
            # VRAIE POSITION
            true_lat = coords_db[query_key]['latitude']
            true_lon = coords_db[query_key]['longitude']

            # PRÉDICTION
            image_path = path_obj / (query_key + ".jpg")
            if not image_path.exists(): continue
            
            results = use_model(r_model, EMBED_PATH, str(image_path))
            if not results: continue

            # On prend le TOP 1 (le premier résultat de usemodel)
            pred_id_raw = results[0][0]
            pred_id_int = int(pred_id_raw.item()) if hasattr(pred_id_raw, 'item') else int(pred_id_raw)
            
            if pred_id_int in id_map:
                cand_key = id_map[pred_id_int]
                pred_lat = coords_db[cand_key]['latitude']
                pred_lon = coords_db[cand_key]['longitude']
                
                dist_err = haversine(true_lat, true_lon, pred_lat, pred_lon)
                color = get_error_color(dist_err)

                # A. Tracer la ligne entre Réel et Prédit
                folium.PolyLine(
                    locations=[(true_lat, true_lon), (pred_lat, pred_lon)],
                    color=color,
                    weight=2,
                    opacity=0.6,
                    tooltip=f"Erreur: {int(dist_err)}m"
                ).add_to(fg_lines)

                # B. Marqueur pour la position RÉELLE (Petit cercle bleu)
                folium.CircleMarker(
                    location=[true_lat, true_lon],
                    radius=3,
                    color='blue',
                    fill=True,
                    popup=f"Réel: {query_key}"
                ).add_to(m)

                # C. Marqueur pour la position PRÉDITE (Icone colorée)
                folium.Marker(
                    location=[pred_lat, pred_lon],
                    icon=folium.Icon(color=color, icon='info-sign'),
                    popup=f"Pred pour {query_key}<br>Erreur: {int(dist_err)}m"
                ).add_to(fg_points)

        except Exception as e:
            continue

    # Ajouter un sélecteur de couches
    folium.LayerControl().add_to(m)
    
    m.save(OUTPUT_FILE)
    print(f"Carte sauvegardée : {OUTPUT_FILE}")

def run_spider_map(r_model, max_points=MAXPOINT, output_file=OUTPUT_FILE, data_dir=DATA_DIR, json_path=JSON_PATH, embed_path=EMBED_PATH):
    global MAXPOINT, EMBED_PATH, JSON_PATH, DATA_DIR, OUTPUT_FILE
    MAXPOINT = max_points
    EMBED_PATH = embed_path
    JSON_PATH = json_path
    DATA_DIR = data_dir
    OUTPUT_FILE = output_file

    run_error_map(r_model)