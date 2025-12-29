import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from model import *
from dataset import *
from tqdm import tqdm
from hierarchical_kmeans import HKMeans

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
    ids = []
    elems = []

    dataset = RoundAboutInferenceDataset(imgs)

    loader = DataLoader(dataset, batch_size=128)

    pbar = tqdm(loader, desc=f"Calcul des embeds", unit="batch", leave=False)
    with torch.no_grad():
        for i_batch, img_batch in pbar:
            img_batch = img_batch.to(DEVICE)
            vecs = model(img_batch).detach().cpu().numpy()

            for idx,vec in zip(i_batch, vecs):
                i = idx.item()

                ids.append(i)
                elems.append(vec)
    
    print("Création des clusters")
    #print(elems)
    ids = np.array(ids)
    elems = np.array(elems)
    db = HKMeans(elems, ids, 5, 10)

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

    vec = vec.detach().cpu().numpy()
    #vec = torch.
    return db.find_elem(vec)

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



