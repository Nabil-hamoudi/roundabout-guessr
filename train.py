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

criterion = nn.TripletMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1,11):
    print("Début de l'epoch " + str(epoch) + " sur 10")

    model.train()
    total_loss = 0
    for img_a, img_b, img_c in train_loader:
        img_a, img_b, img_c = img_a.to(DEVICE), img_b.to(DEVICE), img_c.to(DEVICE)

        #img a = l'ancre, b = l'image actuelle, c = négative
        pred_a = model(img_a)
        pred_b = model(img_b)
        pred_c = model(img_c)

        loss = criterion(pred_a, pred_b, pred_c)

        loss.backward()
        optimizer.step()
        print(loss.item())
        total_loss += loss.item()
    
    print(f"Fin epoch {epoch} loss moyenne {total_loss/len(train_loader)}")
