import json
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import math


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

compat_transform = A.Compose([
    A.Resize(518, 518),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

#on l'utilise encore elle ? c'est trop imprécis
def long_lat_to_km(long,lat):
    #formule : 111.11km = 1° lat
    #          111.11 * cos(lat) = 1° long
    lat_km = 111.11*lat
    long_km = 111.11*math.cos(lat)*long

    return (long_km,lat_km)


def get_images_pos(path):
    with open(path) as f:
        json_file = json.load(f)
    
    coords = []
    # Trier les clés par numéro pour correspondre à get_images_paths
    sorted_keys = sorted(json_file.keys(), 
                        key=lambda k: int(k.split("_")[1]))
    
    for key in sorted_keys:
        data = json_file[key]
        if "latitude" in data and "longitude" in data:
            coords.append((data["latitude"], data["longitude"]))
    
    return coords

def get_images_paths(path="yo/data"):
    files = list(Path(path).glob("*.jpg"))
    #print("alo", path, len(list(files)))

    return sorted(files, key=lambda f: int(f.stem.split("_")[-1]))
class ImagesPosDataset(Dataset):
    def __init__(self, images_paths, images_positions, want_index = False, is_train=False):
        self.images_paths = images_paths
        self.positions = images_positions
        self.want_index = want_index  
        self.is_train = is_train
        self.noise_std = 0.0013  
        self.train_transform = A.Compose([
            A.Resize(518, 518),
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                
                A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            ], p=0.7),
                    
            A.OneOf([
                A.ImageCompression(quality_lower=70, quality_upper=100),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(var_limit=(10.0, 30.0)),
            ], p=0.3), 

            A.CoarseDropout(
                max_holes=4, max_height=32, max_width=32, 
                min_holes=1, min_height=16, min_width=16, 
                fill_value=0, p=0.2
            ),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
            ToTensorV2()
        ])

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
    ds = ImagesPosDataset(imgs_paths, pos)

    for elem in ds:
        print(elem)
    print(pos)