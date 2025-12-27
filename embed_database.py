import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from model import *
from dataset import *

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
    with torch.no_grad():
        for i in range(len(imgs)):
            db[i] = []
            for img in imgs[i]:
                img = torch.from_numpy(img).float().to(DEVICE)
                #Comme on a pas utilisé de loader on doit simuler le batch
                img = img.unsqueeze(0)
                vec = model(img)
                db[i].append(vec.detach().cpu())
                cur_inf += 1
                per = cur_inf/tot
                if math.floor(per*100) % 10 == 0:
                    print(per)

    return db

def get_roundabout(db, img, model):
    model.eval()

    img = torch.from_numpy(img).float().to(DEVICE)
    #Comme on a pas utilisé de loader on doit simuler le batch
    img = img.unsqueeze(0)
    with torch.no_grad():
        
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



