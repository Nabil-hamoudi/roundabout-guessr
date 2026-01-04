import os
import json
import numpy as np
from streetlevel import streetview
import time
import random
from tqdm import tqdm
from PIL import Image
import py360convert

TARGET_IMAGES = 10000
DATA_DIR = "data"
JSON_FILE = "coordinates.json"

EUROPE_BBOX = {
    'lat_min': 36.0,
    'lat_max': 71.0,
    'lon_min': -10.0,
    'lon_max': 30.0
}

DIRECTIONS = [
    (0, 0, "front"),   
    (90, 0, "right"),    
    (180, 0, "back"),   
    (270, 0, "left"),   
]

def sample_random_coordinates_europe(n_samples):
    coords = []
    for _ in range(n_samples):
        lat = np.random.uniform(EUROPE_BBOX['lat_min'], EUROPE_BBOX['lat_max'])
        lon = np.random.uniform(EUROPE_BBOX['lon_min'], EUROPE_BBOX['lon_max'])
        coords.append((lat, lon))
    return coords

def download_street_view_images(coords, output_dir, json_file, max_radius=500, num_views=4):
    os.makedirs(output_dir, exist_ok=True)
    
    coordinates_data = {}
    success_count = 0
    attempt_count = 0
    
    pbar = tqdm(total=TARGET_IMAGES, desc="Downloading images")
    
    for lat, lon in coords:
        if success_count >= TARGET_IMAGES:
            break
        
        attempt_count += 1
        
        try:
            pano = streetview.find_panorama(lat, lon, radius=max_radius)
            
            if pano is None:
                continue
            
            panorama_img = streetview.get_panorama(pano, zoom=3)
            pano_array = np.array(panorama_img)
            
            for h_deg, v_deg, direction in DIRECTIONS[:num_views]:
                if success_count >= TARGET_IMAGES:
                    break
                
                perspective_array = py360convert.e2p(
                    pano_array,
                    fov_deg=(90, 90), 
                    u_deg=h_deg,   
                    v_deg=v_deg,  
                    out_hw=(1080, 1920),
                    mode='bilinear'
                )
                
                image_id = f"img_{success_count:05d}"
                image_path = os.path.join(output_dir, f"{image_id}.jpg")
                
                perspective_img = Image.fromarray(perspective_array.astype(np.uint8))
                perspective_img.save(image_path, quality=90)
                
                coordinates_data[image_id] = {
                    "longitude": pano.lon,
                    "latitude": pano.lat,
                    "pano_id": pano.id,
                    "sampled_lon": lon,
                    "sampled_lat": lat,
                    "view_direction": direction,
                    "h_deg": h_deg,
                    "v_deg": v_deg
                }
                
                success_count += 1
                pbar.update(1)

            time.sleep(random.uniform(0.1, 0.3))
            
            if success_count % 100 == 0:
                with open(json_file, 'w') as f:
                    json.dump(coordinates_data, f, indent=2)
            
        except Exception as e:
            continue
    
    pbar.close()
    
    with open(json_file, 'w') as f:
        json.dump(coordinates_data, f, indent=2)
    
    print(f"\nTentatives: {attempt_count}, Succès: {success_count}")
    print(f"Taux de réussite: {success_count/attempt_count*100:.1f}%")
    
    return success_count

def main():
    print("=== Script de téléchargement d'images Street View en Europe ===")
    print(f"Objectif: {TARGET_IMAGES} images")
    print(f"Zone: Europe entière (bbox: {EUROPE_BBOX})")
    print(f"Format: Vues perspective (1920x1080) - 4 DIRECTIONS par panorama")
    
    n_panoramas_needed = (TARGET_IMAGES // 4) + 100
    n_samples = n_panoramas_needed * 10
    
    print(f"\nSampling de {n_samples} coordonnées aléatoires...")
    coords = sample_random_coordinates_europe(n_samples)
    
    print(f"\nTéléchargement des images Street View...")
    
    success_count = download_street_view_images(coords, DATA_DIR, JSON_FILE, num_views=4)
    
    print(f"\n=== Terminé ===")
    print(f"Images téléchargées avec succès: {success_count}")
    print(f"Dossier: {DATA_DIR}")
    print(f"Fichier JSON: {JSON_FILE}")

if __name__ == "__main__":
    main()
