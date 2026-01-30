import os
import json
import numpy as np
try:
    from streetlevel import streetview
    import py360convert
except:
    print("You need streetlevel and py360convert to run scraping")
    print("run 'pip install streetlevel py360convert'")
import time
import random
from tqdm import tqdm
from PIL import Image


#nb d'images voulues
TARGET_IMAGES = 1000 
DATA_DIR = "test_set_paris"
JSON_FILE = "coo_test_set_paris.json"

#bbox paris++
LAT_MIN, LAT_MAX = 48.77, 48.97
LON_MIN, LON_MAX = 2.22, 2.47

DIRECTIONS = [
    (0, 0, "front"),   
    (180, 0, "back"),   
]

def sample_uniform_paris(n_samples):
    coords = []
    print(f"Génération de coordonnées uniformes sur Lat[{LAT_MIN}-{LAT_MAX}] / Lon[{LON_MIN}-{LON_MAX}]")
    
    for _ in range(n_samples):
        lat = np.random.uniform(LAT_MIN, LAT_MAX)
        lon = np.random.uniform(LON_MIN, LON_MAX)
        coords.append((lat, lon))
        
    return coords

def download_street_view_images(coords, output_dir, json_file, max_radius=50, num_views=2):
    os.makedirs(output_dir, exist_ok=True)
    
    #on reprend pas de 0 si on a déjà commencé
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                coordinates_data = json.load(f)
        except:
            coordinates_data = {}
    else:
        coordinates_data = {}
        
    success_count = len(coordinates_data)
    #on part du principe qu'il n'y a pas de trous
    #ce qui est généralement le cas si le json sort de ce script
    current_idx = success_count 

    attempt_count = 0
    consecutive_failures = 0
    
    pbar = tqdm(total=TARGET_IMAGES, initial=success_count, desc="DL")
    
    for lat, lon in coords:
        if success_count >= TARGET_IMAGES:
            break
        
        attempt_count += 1
        
        try:

            pano = streetview.find_panorama(lat, lon, radius=max_radius)
            
            if pano is None:
                consecutive_failures += 1
                continue
            
            consecutive_failures = 0
            
            if any(p.get('pano_id') == pano.id for p in coordinates_data.values()):
                continue

            panorama_img = streetview.get_panorama(pano, zoom=3)
            if panorama_img is None:
                continue
                
            pano_array = np.array(panorama_img)
            batch_success = True
            temp_metadata = {}
            
            for h_deg, v_deg, direction in DIRECTIONS[:num_views]:
                if success_count >= TARGET_IMAGES:
                    break
                
                try:
                    perspective_array = py360convert.e2p(
                        pano_array,
                        fov_deg=(90, 90), 
                        u_deg=h_deg,   
                        v_deg=v_deg,  
                        out_hw=(518, 518), #on va resize à cette size donc autant pas prendre trop de place
                        mode='bilinear'
                    )
                    
                    image_id = f"paris_{current_idx:06d}" # 6 chiffres pour supporter >100k
                    image_path = os.path.join(output_dir, f"{image_id}.jpg")
                    
                    perspective_img = Image.fromarray(perspective_array.astype(np.uint8))
                    perspective_img.save(image_path, quality=85)
                    
                    #on met la metadata historique
                    #en réalité ça sert pas trop
                    #mais on avait l'intention de l'utiliser
                    temp_metadata[image_id] = {
                        "file_name": f"{image_id}.jpg",
                        "longitude": pano.lon,
                        "latitude": pano.lat,
                        "pano_id": pano.id,
                        "sampled_lon": lon,
                        "sampled_lat": lat,
                        "view_direction": direction,
                        "date": str(pano.date) if pano.date else None
                    }
                    current_idx += 1
                    success_count += 1
                    pbar.update(1)
                except Exception as e:
                    batch_success = False
                    break
            
            if batch_success:
                coordinates_data.update(temp_metadata)
            
            #pair ou impair ? en fonction de si il y a eu erreur !
            if success_count % 50 == 0 or success_count % 50 == 1:
                with open(json_file, 'w') as f:
                    json.dump(coordinates_data, f, indent=2)
            
            #petite pause pour pas trop spam
            #--> Nabil encore ban à ce jour
            # RIP
            time.sleep(random.uniform(0.1, 0.3))
            
        except Exception as e:
            continue
    
    pbar.close()
    
    with open(json_file, 'w') as f:
        json.dump(coordinates_data, f, indent=2)
    
    print(f"Images sauvegardées: {success_count}")
    
    return success_count

def main():

    #on prend large, si jamais y'a des timeouts
    #ou des positions trop peu fertiles (devrait pas arriver à paris, mais en france ou europe si)
    n_panoramas_needed = (TARGET_IMAGES // 2) + 100
    n_samples = n_panoramas_needed * 3 
    
    print(f"\nGénération de {n_samples} coordonnées uniformes...")
    coords = sample_uniform_paris(n_samples)
    
    print(f"Lancement du téléchargement dans '{DATA_DIR}'...")
    download_street_view_images(coords, DATA_DIR, JSON_FILE, max_radius=50, num_views=2)
    

if __name__ == "__main__":
    main()