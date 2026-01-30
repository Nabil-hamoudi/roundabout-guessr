import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.base.dataset import *
from tqdm import tqdm
import cv2

LAT_MIN_F, LAT_MAX_F = 41.3, 51.1
LON_MIN_F, LON_MAX_F = -5.1, 9.6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
def create_database(imgs, pos, model):

    print("Creating the database")
    model.eval()
    """
    Le format là en gros c'est 
        roundabout : [liste des vecteurs d'embed]

    -> C'est super long, on verra plus tard pour opti.
    -> ça a l'avantage d'être super simple !
    """
    ids = []
    elems = []

    dataset = ImagesPosDataset(imgs, pos, want_index=True)
    loader = DataLoader(dataset, batch_size=32)

    pbar = tqdm(loader, desc=f"Calcul des embeds", unit="batch", leave=False)
    with torch.no_grad():
        for img, pos, i_batch in pbar:
            img = img.to(DEVICE)
            #pos = pos.to(DEVICE)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                vecs = model.image_encoder(img).detach().cpu().numpy()

            for idx,vec in zip(i_batch, vecs):
                i = idx.item()

                ids.append(i)
                elems.append(vec)
    a = {
        "ids" : ids,
        "elems" : elems
    }
    torch.save(a, 'embeddings_db.pt')

    #print(elems)
    return a

def load_database(path='embeddings_db.pt'):
    return torch.load(path, weights_only=False)

def get_closest(db, img, model, k = 5):
    model.eval()
    
    img_tensor = compat_transform(image=img)["image"]
    img_tensor = img_tensor.to(DEVICE).unsqueeze(0) 

    with torch.no_grad():
        query_vec = model.image_encoder(img_tensor)
        query_vec = F.normalize(query_vec, p=2, dim=1)

    db_vecs = torch.tensor(np.array(db['elems']), device=DEVICE)
    db_ids = np.array(db['ids'])

    #query x dbvect.T
    scores = torch.mm(query_vec, db_vecs.T)
    
    best_scores, best_indices = torch.topk(scores, k=k, dim=1)
    
    best_indices = best_indices.cpu().numpy()[0]
    best_scores = best_scores.cpu().numpy()[0]
    
    results = []
    for rank, idx in enumerate(best_indices):
        result_id = db_ids[idx]
        score = best_scores[rank]
        results.append((result_id, score))
        
    return results


def create_embeddings(model, data_path, json_path):
    pos = get_images_pos(json_path)
    imgs = get_images_paths(data_path)

    db = create_database(imgs, pos, model)
    torch.save(db, "embeddings_db.pt")


def tsne(embeding_path, json_path):
    pos = get_images_pos(json_path)
    visualize_tsne(embeding_path, pos_dict=pos)


def pca(embeding_path):
    visualize_pca(embeding_path)

def pca_geo(embeding_path, json_path):
    pos = get_images_pos(json_path)
    compare_pca_geo(embeding_path, pos_dict=pos)

def use_model(model, embeding_path, image_path):
    a = torch.load(embeding_path, weights_only=False)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return get_closest(a, img, model)

def get_closest_locations(model, embeding_path, image_path, pos):
    results = use_model(model, embeding_path, image_path)
    pos = get_images_pos(pos)
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="guessr_amis_project")
    for rank, (idx, score) in enumerate(results):
        idx = int(idx)

        try:
            lat, lon = pos[idx]
            
            try:
                location = geolocator.reverse(f"{lat}, {lon}", language='fr', exactly_one=True)
                address = location.address if location else "Adresse inconnue"
            except Exception as e:
                address = f"Erreur de géocodage : {e}"

            print(f"Rang {rank+1}")
            print(f"Similarité : {score:.4f} (Confiance)")
            print(f"GPS        : {lat:.6f}, {lon:.6f}")
            print(f"Adresse    : {address}")
            
            # Lien Google Maps cliquable pour frimer
            gmaps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            print(f"Google Maps: {gmaps_link}")
            print("-" * 60)
            
            time.sleep(1)
            
        except KeyError:
            print(f"ID {idx} non trouvé dans le fichier JSON des coordonnées.")

def init_model(model_path, cross, france):
    if cross:
        from src.cross_view import model_cross
        if france:
            r_model = model_cross.CrossEncoder(LAT_MIN=LAT_MIN_F, LAT_MAX=LAT_MAX_F, LON_MIN=LON_MIN_F, LON_MAX=LON_MAX_F).to(DEVICE)
        else:
            r_model = model_cross.CrossEncoder().to(DEVICE)
    else:
        from src.base import model
        if france:
            r_model = model.MixedEncoder(LAT_MIN=LAT_MIN_F, LAT_MAX=LAT_MAX_F, LON_MIN=LON_MIN_F, LON_MAX=LON_MAX_F).to(DEVICE)
        else: 
            r_model = model.MixedEncoder().to(DEVICE)
    r_model.load_state_dict(torch.load(model_path))
    return r_model
