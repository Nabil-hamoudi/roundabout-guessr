import json
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import random
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

compat_transform = A.Compose([
    A.Resize(400, 600),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

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

def load_images(nb_roundabouts, path = "data"):
    # la fonction part du principe qu'on est en jpg par contre
    all_images = []
    for roundabout_id in range(1,nb_roundabouts+1):
        roundabout_folder = path + "/roundabout_"+str(roundabout_id)
        files = sorted(Path(roundabout_folder).glob("*.jpg"), key=lambda f: int(f.name.replace("streetview_","").rstrip(".jpg")))
        all_images.append([cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files])
    return all_images

class RoundAboutTrainingDataset(Dataset):
    def __init__(self, roundabouts, roundabouts_positions):
        # Roundabouts liste de liste de liste d'images
        # ATTENTION c vite vachement lourd en mémoire
        # ça ira pour l'instant, mais si jamais il faut à tout pris
        # qu'on le change

        # Roundabout_positions c'est liste une liste de tuples de pos

        if len(roundabouts) != len(roundabouts_positions):
            print("On handle pas les éléments non labelés ça va crash !")
            return

        #Si on a malheureusement pas d'image compatible
        #En vrai on devra sûrement faire ça pour toutes les images
        #à voir !!
        self.augment_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        ])

        #Liste de tuples (img, pos)
        #Pour le metric learning il faut (img1,img2,img3)
        self.roundabouts = roundabouts
        self.elems = []

        for i in range(len(roundabouts)):
            for j in range(len(roundabouts[i])):
                img_a = roundabouts[i][j]
                self.elems.append((i, img_a))

    def __len__(self):
        return len(self.elems)
    
    def __getitem__(self, index):
        i, img_a = self.elems[index]

        if len(self.roundabouts[i]) > 1:
            img_b = random.choice([x for x in self.roundabouts[i] if id(x) != id(img_a)])
        else:
            img_b = self.augment_transform(image = img_a)["image"]
        other_idx = random.choice([x for j,x in enumerate(range(len(self.roundabouts))) if j!=i and len(self.roundabouts[j]) != 0])
        img_c = random.choice(self.roundabouts[other_idx])

        #img_a,img_b,img_c = self.elems[index]
        img_a = compat_transform(image = img_a)["image"]
        img_b = compat_transform(image = img_b)["image"]
        img_c = compat_transform(image = img_c)["image"]
        return img_a,img_b,img_c
        #return torch.from_numpy(img).float(), torch.from_numpy(np.asarray(pos, np.float32))

class RoundAboutInferenceDataset(Dataset):
    def __init__(self, roundabouts):
        self.roundabouts = roundabouts
        self.roundabouts = roundabouts
        self.elems = []

        for i in range(len(roundabouts)):
            for j in range(len(roundabouts[i])):
                img_a = roundabouts[i][j]
                self.elems.append((i, img_a))
    def __len__(self):
        return len(self.roundabouts)
    
    def __getitem__(self, index):
        i, img = self.elems[index]
        return i, torch.from_numpy(img).float()

if __name__ == "__main__":
    pos = get_roundabouts_pos("roundabouts.json")
    imgs = load_images(len(pos))
    ds = RoundAboutTrainingDataset(imgs, pos)

    for elem in ds:
        pass
        #print(elem)
    print(pos)