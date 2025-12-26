import json
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_roundabouts_pos(path):
    with open(path) as f:
        json_file = json.load(f)
    #On compte sur le fait que ce soit dans l'ordre
    coords = []
    for elem in json_file["features"]:
        #devrait pas arriver mais bon
        if not "geometry" in elem: continue
        if not "coordinates" in elem["geometry"]: continue
        #Hardcodé mgl
        if elem["geometry"]["type"] != "LineString" :
            coords.append([elem["geometry"]["coordinates"]])
            continue
        roundabout_coords = []
        for coord in elem["geometry"]["coordinates"]:
            roundabout_coords.append(coord)
        #On met le centre de chaque image !! utile pour embed
        coords.append(roundabout_coords)
    
    return coords

def load_images(nb_roundabouts):
    # la fonction part du principe qu'on est en jpg par contre
    all_images = []
    for roundabout_id in range(1,nb_roundabouts+1):
        roundabout_folder = "data/roundabout_"+str(roundabout_id)
        files = sorted(Path(roundabout_folder).glob("*.jpg"), key=lambda f: int(f.name.replace("streetview_","").rstrip(".jpg")))
        all_images.append([cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files])
    return all_images

class RoundAboutDataset(Dataset):
    def __init__(self, roundabouts, roundabouts_positions):
        # Roundabouts liste de liste de liste d'images
        # ATTENTION c vite vachement lourd en mémoire
        # ça ira pour l'instant, mais si jamais il faut à tout pris
        # qu'on le change

        # Roundabout_positions c'est liste une liste de tuples de pos

        if len(roundabouts) != len(roundabouts_positions):
            print("On handle pas les éléments non labelés ça va crash !")
            return

        #Liste de tuples (img, pos)
        self.elems = []

        for i in range(len(roundabouts)):
            for j in range(len(roundabouts[i])):
                self.elems.append((roundabouts[i][j], roundabouts_positions[i][j]))

    def __len__(self):
        return len(self.elems)
    
    def __getitem__(self, index):
        img, pos = self.elems[index]
        img = (img-IMAGENET_MEAN)/IMAGENET_STD
        return torch.from_numpy(img).float(), torch.from_numpy(np.asarray(pos, np.float32))
    
if __name__ == "__main__":
    pos = get_roundabouts_pos("roundabouts.json")
    imgs = load_images(len(pos))
    ds = RoundAboutDataset(imgs, pos)

    for elem in ds:
        print(elem)
    print(pos)