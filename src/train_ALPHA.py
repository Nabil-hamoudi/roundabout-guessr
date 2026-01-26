import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from dataset import *
from model_ALPHA import *
from embed_database import *
import random
from torch.optim.lr_scheduler import LinearLR
from loss import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_validation(model, val_imgs, criterion):
    model.eval()
    
    total_loss = 0
    pbar = tqdm(val_imgs, desc=f"Validation", unit="query", leave=False)

    with torch.no_grad():
        for img_a, img_b, pos_a in pbar:
            img_a, img_b = img_a.to(DEVICE), img_b.to(DEVICE)
            pos_a = pos_a.to(DEVICE)
            #img a = l'ancre, b = l'image actuelle, c = négative
            pred_a = model(img_a)
            pred_b = model(img_b)

            loss = criterion(pred_a, pred_b, pos_a)
            #print(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()

    return total_loss/len(val_imgs)

if __name__ == "__main__":

    print("Chargement du JSON rond pts")
    pos = get_images_pos("yo/coordinates.json")
    print("Chargement des images")
    imgs = get_images_paths()


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
    model.load_state_dict(torch.load("save.pt"))

    #Gérer la validation après déjà je veux faire en sorte que ça forward
    train_loader = DataLoader(train_dataset, batch_size=24, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, num_workers=8)
    criterion = BatchHardLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in range(1,101):
        print("Début de l'epoch " + str(epoch) + " sur 10")

        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)

        for img_a, img_b, pos_a in pbar:
            img_a, img_b= img_a.to(DEVICE), img_b.to(DEVICE)
            pos_a = pos_a.to(DEVICE)
            optimizer.zero_grad()
            #img a = l'ancre, b = l'image actuelle, c = négative
            pred_a = model(img_a)
            pred_b = model(img_b)

            loss = criterion(pred_a, pred_b, pos_a)

            loss.backward()
            optimizer.step()
            #print(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()
            #break
        print(f"Fin epoch {epoch} loss tr moyenne {total_loss/len(train_loader)}")
        print("Début de la validation : ")
        loss = model_validation(model, val_loader, criterion)
        scheduler.step(loss)
        print(f"Validation terminée, loss : {loss}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

        
