import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from dataset import *
from model import *
from embed_database import *
import random

from loss import GeoTripletLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Chargement du JSON rond pts")
pos = get_images_pos("yo/coordinates.json")
print("Chargement des images")
imgs = get_images_paths()


def model_validation(model, val_imgs, criterion):
    model.eval()
    
    total_loss = 0
    pbar = tqdm(val_imgs, desc=f"Validation", unit="query", leave=False)

    with torch.no_grad():
        for img_a, img_b, img_c, pos_a, pos_c in pbar:
            img_a, img_b, img_c = img_a.to(DEVICE), img_b.to(DEVICE), img_c.to(DEVICE)
            pos_a, pos_c = pos_a.to(DEVICE), pos_c.to(DEVICE)
            #img a = l'ancre, b = l'image actuelle, c = négative
            pred_a = model(img_a)
            pred_b = model(img_b)
            pred_c = model(img_c)

            loss = criterion(pred_a, pred_b, pred_c, pos_a, pos_c)
            #print(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()

    return total_loss/len(val_imgs)
dataset = ImagesTrainingDataset(imgs, pos)
generator1 = torch.Generator().manual_seed(42)
train_indices, val_indices = torch.utils.data.random_split(
    range(len(dataset)), 
    [0.95, 0.05], 
    generator=generator1
)

train_dataset = Subset(dataset, train_indices.indices)
val_dataset = Subset(dataset, val_indices.indices)

model = BaseEmbed().to(DEVICE)

#Gérer la validation après déjà je veux faire en sorte que ça forward
train_loader = DataLoader(train_dataset, batch_size=2)
val_loader = DataLoader(val_dataset, batch_size=2)
criterion = GeoTripletLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1,11):
    print("Début de l'epoch " + str(epoch) + " sur 10")

    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)

    for img_a, img_b, img_c, pos_a, pos_c in pbar:
        img_a, img_b, img_c = img_a.to(DEVICE), img_b.to(DEVICE), img_c.to(DEVICE)
        pos_a, pos_c = pos_a.to(DEVICE), pos_c.to(DEVICE)
        optimizer.zero_grad()
        #img a = l'ancre, b = l'image actuelle, c = négative
        pred_a = model(img_a)
        pred_b = model(img_b)
        pred_c = model(img_c)

        loss = criterion(pred_a, pred_b, pred_c, pos_a, pos_c)

        loss.backward()
        optimizer.step()
        #print(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        total_loss += loss.item()
        #break
    print(f"Fin epoch {epoch} loss tr moyenne {total_loss/len(train_loader)}")
    print("Début de la validation : ")
    loss = model_validation(model, val_loader, criterion)
    print(f"Validation terminée, loss : {loss}")

    
