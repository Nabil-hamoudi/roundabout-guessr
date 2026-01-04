import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from model import *
from dataset import *
from tqdm import tqdm
from hierarchical_kmeans import HKMeans
import cv2
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

    dataset = ImagesInferenceDataset(imgs)

    loader = DataLoader(dataset, batch_size=16)

    pbar = tqdm(loader, desc=f"Calcul des embeds", unit="batch", leave=False)
    with torch.no_grad():
        for i_batch, img_batch in pbar:
            img_batch = img_batch.to(DEVICE)
            vecs = model(img_batch).detach().cpu().numpy()

            for idx,vec in zip(i_batch, vecs):
                i = idx.item()

                ids.append(i)
                elems.append(vec)
    a = {
        "ids" : ids,
        "elems" : elems
    }
    torch.save(a, 'embeddings_db.pt')
    print("Création des clusters")
    #print(elems)
    ids = np.array(ids)
    elems = np.array(elems)
    db = HKMeans(elems, ids, 5, 10)

    return db

def get_closest(db, img, model):
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
    pos = get_images_pos("yo/coordinates.json")
    imgs = get_images_paths()

    model = BaseEmbed().to(DEVICE)
    #On peut aussi db = torch.load(embeddings_db.pt)
    #db = create_database(imgs, model)
    #db = torch.load("embeddings_db.pt")
    #torch.save(db, "embeddings_db.pt")

    #a = torch.load("embeddings_db.pt", weights_only=False)
    #ids = np.array(a["ids"])
    #elems = np.array(a["elems"])
    db = torch.load("embeddings_db.pt", weights_only=False)
    #model = BaseEmbed().to(DEVICE)   
    
    
    img = cv2.imread("./val/roundabout_263/streetview_1.jpg", cv2.IMREAD_COLOR)

    print(get_closest(db, img, model))



