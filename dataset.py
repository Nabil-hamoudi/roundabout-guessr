import json
from matplotlib import transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import random
import math
from sklearn.neighbors import BallTree


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

compat_transform = A.Compose([
    A.Resize(700, 1274),
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
            A.RandomResizedCrop(size=(720, 1280), scale=(0.3, 1.0), p=1.0),
            A.ToGray(p=0.2),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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

        # Charger images
        img_b = cv2.imread(str(self.images_paths[index]), cv2.IMREAD_COLOR)
        #img_c = cv2.imread(str(self.images_paths[idx_neg]), cv2.IMREAD_COLOR)
        #pos_c = self.positions[idx_neg]
        
        # Transforms
        img_a = self.train_transform(image=img_a)["image"]
        img_b = self.train_transform(image=img_b)["image"]
        #img_c = self.train_transform(image=img_c)["image"]
        
        pos_a = torch.tensor(pos_a, dtype=torch.float32)
        #pos_c = torch.tensor(pos_c, dtype=torch.float32)
        
        return img_a, img_b, pos_a
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
    
class ImagesPosDataset(Dataset):
    def __init__(self, images_paths, images_positions, want_index = False, is_train=False):
        self.images_paths = images_paths
        self.positions = images_positions
        self.want_index = want_index  
        self.is_train = is_train
        self.noise_std = 0.0001
        self.train_transform = A.Compose([
            A.Resize(700, 1274),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            ], p=0.8),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
            ], p=0.2),

            A.CoarseDropout(
                max_holes=8, max_height=16, max_width=16, 
                min_holes=1, min_height=8, min_width=8, 
                fill_value=0, p=0.2
            ),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
            ToTensorV2()
        ])

        #pas d'augmentations pour l'instant on regarde après !!

    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, index):
        img = cv2.imread(str(self.images_paths[index]), cv2.IMREAD_COLOR)
        if self.is_train:
            img = self.train_transform(image=img)["image"]
        else:
            img = compat_transform(image=img)["image"]

        raw_pos = self.positions[index]
        
        lat = raw_pos[0]
        lon = raw_pos[1]
        #print(self.is_train)
        #ajout bruit gaussien
        if self.is_train:
            lat += np.random.randn() * self.noise_std
            lon += np.random.randn() * self.noise_std
            
            #on reste dans les bornes
            lat = np.clip(lat, -90, 90)
            lon = np.clip(lon, -180, 180)

        pos = torch.tensor([lat, lon], dtype=torch.float32)
        if self.want_index:
            return img, pos, index
        return img, pos

if __name__ == "__main__":
    pos = get_images_pos("yo/coordinates.json")
    imgs_paths = get_images_paths()
    ds = ImagesTrainingDataset(imgs_paths, pos)

    for elem in ds:
        print(elem)
    print(pos)