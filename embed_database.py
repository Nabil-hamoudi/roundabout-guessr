import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from model import *
from dataset import *
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_database(imgs, model):

    print("Creating the databse")
    model.eval()
    """
    Le format là en gros c'est 
        roundabout : [liste des vecteurs d'embed]

    -> C'est super long, on verra plus tard pour opti.
    -> ça a l'avantage d'être super simple !
    """
    db = {}
    cur_inf = 0
    tot = 9351 #j'hardcode oui et alors

    dataset = RoundAboutInferenceDataset(imgs)

    loader = DataLoader(dataset, batch_size=128)

    pbar = tqdm(loader, desc=f"Création", unit="batch", leave=False)
    with torch.no_grad():
        for i_batch, img_batch in pbar:

            img_batch = img_batch.to(DEVICE)

            vecs = model(img_batch).detach().cpu()

            for idx,vec in zip(i_batch, vecs):
                i = idx.item()

                if i not in db:
                    db[i] = []
                
                db[i].append(vec.unsqueeze(0))
    return db

def get_roundabout(db, img, model):
    model.eval()
    img = compat_transform(image = img)["image"]
    img = img.to(DEVICE)
    #Comme on a pas utilisé de loader on doit simuler le batch
    img = img.unsqueeze(0)
    with torch.no_grad():
        #img = compat_transform(image = img)
        vec = model(img)

    closest_roundabout = -1
    min_dist = float('inf')

    for roundabout_id in db:
        roundabout_vectors = db[roundabout_id]
        
        for r_vec in roundabout_vectors:
            r_vec = r_vec.to(DEVICE)

            dist = torch.norm(vec - r_vec)
            
            if dist < min_dist:
                closest_roundabout = roundabout_id
                min_dist = dist
    
    return closest_roundabout

if __name__ == "__main__":
    pos = get_roundabouts_pos("roundabouts.json")
    imgs = load_images(len(pos))

    model = BaseEmbed().to(DEVICE)
    #On peut aussi db = torch.load(embeddings_db.pt)
    db = create_database(imgs, model)
    #db = torch.load("embeddings_db.pt")
    torch.save(db, "embeddings_db.pt")
        
    imgs_val = load_images(len(pos), "val")
    img = imgs_val[262][0]

    print(get_roundabout(db, img, model))



