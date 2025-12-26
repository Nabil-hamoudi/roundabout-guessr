import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import *
from model import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Chargement du JSON rond pts")
pos = get_roundabouts_pos("roundabouts.json")
print("Chargement des images")
imgs = load_images(len(pos))

dataset = RoundAboutDataset(imgs, pos)

model = BaseEmbed().to(DEVICE)

#Gérer la validation après déjà je veux faire en sorte que ça forward
train_loader = DataLoader(dataset)

for epoch in range(1,11):
    print("Début de l'epoch " + str(epoch) + " sur 10")

    model.train()

    for img, pos in train_loader:
        img, pos = img.to(DEVICE), pos.to(DEVICE)

        pred = model(img)