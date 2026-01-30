import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import math
from tqdm import tqdm
from pathlib import Path
import concurrent.futures


#endroit où les coo sont stockées
JSON_PATH = "coo_test_set_paris.json"
#endroit où les sat seront stockées 
OUTPUT_DIR = "sat_test_set_paris"

IMAGE_SIZE = 512
ZOOM_LEVEL_METERS = 300 
MAX_WORKERS = 3   

IGN_WMS_URL = "https://data.geopf.fr/wms-r/wms"

retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

def get_bbox(lat, lon, size_meters):
    R = 6378137
    dn = size_meters / 2.0
    de = size_meters / 2.0
    dLat = dn / R
    dLon = de / (R * math.cos(math.pi * lat / 180))
    
    lat_min = lat - (dLat * 180 / math.pi)
    lat_max = lat + (dLat * 180 / math.pi)
    lon_min = lon - (dLon * 180 / math.pi)
    lon_max = lon + (dLon * 180 / math.pi)
    
    return f"{lat_min},{lon_min},{lat_max},{lon_max}"

def download_one_image(item):
    key, coords = item

    filename = f"sat_{key}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)

    #si déjà dl on redl pas
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return 0 
    
    lat = coords['latitude']
    lon = coords['longitude']
    bbox = get_bbox(lat, lon, ZOOM_LEVEL_METERS)
    
    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.3.0',         
        'REQUEST': 'GetMap',
        'LAYERS': 'ORTHOIMAGERY.ORTHOPHOTOS',
        'STYLES': '',
        'FORMAT': 'image/jpeg',
        'CRS': 'EPSG:4326',    
        'BBOX': bbox,            
        'WIDTH': IMAGE_SIZE,
        'HEIGHT': IMAGE_SIZE
    }
    
    try:
        response = session.get(IGN_WMS_URL, params=params, timeout=10)
        #print(response.status_code)
        if response.status_code == 200 and response.headers.get('Content-Type') == 'image/jpeg':
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return 1
    except Exception:
        pass 
    
    return -1

def main():
    if not os.path.exists(JSON_PATH):
        print(f"No coordonates file found at {JSON_PATH}")
        return
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    items = list(data.items())
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(download_one_image, items), total=len(items), unit="img"))


if __name__ == "__main__":
    main()