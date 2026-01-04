import json
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import random
import math
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

compat_transform = A.Compose([
    A.Resize(720, 1280),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

def long_lat_to_km(long,lat):
    #formule : 111.11km = 1° lat
    #          111.11 * cos(lat) = 1° long
    lat_km = 111.11*lat
    long_km = 111.11*math.cos(lat)*long

    return (long_km,lat_km)


def get_images_pos(path):
    #On compte sur le fait que ce soit dans l'ordre
    with open(path) as f:
        json_file = json.load(f)
    
    coords = []
    
    for data in json_file.values():
        if "latitude" in data and "longitude" in data:
            coords.append((data["latitude"], data["longitude"]))
    
    return coords

def get_images_paths(path="yo/data"):
    files = sorted(Path(path).glob("img_*.jpg"), key=lambda f: int(f.stem.split("_")[1]))
    return files

class ImagesTrainingDataset(Dataset):
    def __init__(self, images_paths, images_positions):
        # images liste de liste de liste d'images
        # ATTENTION c vite vachement lourd en mémoire
        # ça ira pour l'instant, mais si jamais il faut à tout pris
        # qu'on le change

        # images_positions c'est liste une liste de tuples de pos

        if len(images_paths) != len(images_positions):
            print("On handle pas les éléments non labelés ça va crash !")
            return

        #Si on a malheureusement pas d'image compatible
        #En vrai on devra sûrement faire ça pour toutes les images
        #à voir !!
        self.augment_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        ])

        self.train_transform = A.Compose([
            A.RandomResizedCrop(size=(720, 1280), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
            ], p=0.8),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        #Liste de tuples (img, pos)
        #Pour le metric learning il faut (img1,img2,img3)
        self.images_paths = images_paths
        self.positions = images_positions

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
        img_a = cv2.imread(str(self.images_paths[index]), cv2.IMREAD_COLOR)
        pos_a = self.positions[index]

        other_indices = [i for i in range(len(self.images_paths)) if i != index]
        idx_1,idx_2 = random.sample(other_indices, 2)

        img_1 = cv2.imread(str(self.images_paths[idx_1]), cv2.IMREAD_COLOR)
        img_2 = cv2.imread(str(self.images_paths[idx_2]), cv2.IMREAD_COLOR)
        pos_1 = self.positions[idx_1]
        pos_2 = self.positions[idx_2]

        pos_a_km = long_lat_to_km(*pos_a)
        pos_1_km = long_lat_to_km(*pos_1)
        pos_2_km = long_lat_to_km(*pos_2)
        
        dist_1 = np.linalg.norm(np.array(pos_a_km) - np.array(pos_1_km))
        dist_2 = np.linalg.norm(np.array(pos_a_km) - np.array(pos_2_km))

        if dist_1 < dist_2:
            img_b, pos_b = img_1, pos_1
            img_c, pos_c = img_2, pos_2
        else:
            img_b, pos_b = img_2, pos_2
            img_c, pos_c = img_1, pos_1
        #img_a,img_b,img_c = self.elems[index]
        img_a = self.train_transform(image = img_a)["image"]
        img_b = self.train_transform(image = img_b)["image"]
        img_c = self.train_transform(image = img_c)["image"]

        pos_a = torch.tensor(pos_a, dtype = torch.float32)
        pos_c = torch.tensor(pos_c, dtype = torch.float32)

        return img_a,img_b,img_c, pos_a, pos_c
        #return torch.from_numpy(img).float(), torch.from_numpy(np.asarray(pos, np.float32))

class ImagesInferenceDataset(Dataset):
    def __init__(self, images_paths):
        self.elems = images_paths
    def __len__(self):
        return len(self.elems)
    
    def __getitem__(self, index):
        img = cv2.imread(str(self.elems[index]), cv2.IMREAD_COLOR)
        img = compat_transform(image=img)["image"]
        return index, img

if __name__ == "__main__":
    pos = get_images_pos("yo/coordinates.json")
    imgs_paths = get_images_paths()
    ds = ImagesTrainingDataset(imgs_paths, pos)

    for elem in ds:
        print(elem)
    print(pos)