import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from dataset import *
from model2 import *
from embed_database import *
import random
from torch.optim.lr_scheduler import LinearLR
from loss import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
NBR_EPOCH = 101

BATCH_SIZE = 32
BATCH_COMBINED = 4096

def model_validation(model, val_imgs, criterion):
    model.eval()
    
    total_loss = 0
    pbar = tqdm(val_imgs, desc=f"Validation", unit="query", leave=False)

    with torch.no_grad():
        for img, pos in pbar:
            img = img.to(DEVICE)
            pos = pos.to(DEVICE)
            B = img.size(0)
            #img a = l'ancre, b = l'image actuelle, c = négative
            pred_img, pred_pos, scale = model(img, pos)

            logits = (pred_img @ pred_pos.T)*scale
            targets = torch.arange(B, device=DEVICE)

            loss_i2p = criterion(logits, targets)
            loss_p2i = criterion(logits.T, targets)
            loss = (loss_i2p + loss_p2i) / 2

            #print(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()

    return total_loss/len(val_imgs)
import copy 
if __name__ == "__main__":
    scaler = torch.amp.GradScaler(DEVICE_STR)
    print("Chargement du JSON rond pts")
    pos = get_images_pos("yo/coordinates.json")
    print("Chargement des images")
    imgs = get_images_paths()


    dataset = ImagesPosDataset(imgs, pos, is_train=True)
    generator1 = torch.Generator().manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(dataset)), 
        [0.99, 0.01], 
        generator=generator1
    )

    train_dataset = Subset(dataset, train_indices.indices)
    val_dataset = Subset(copy.deepcopy(dataset), val_indices.indices)
    val_dataset.dataset.is_train = False

    print(train_dataset.dataset.is_train)
    print(val_dataset.dataset.is_train)

    model = MixedEncoder().to(DEVICE)
    #model.load_state_dict(torch.load("save.pt"))

    #Gérer la validation après déjà je veux faire en sorte que ça forward
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    criterion = nn.CrossEntropyLoss()#BatchHardLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in range(1, NBR_EPOCH):
        print("Début de l'epoch " + str(epoch) + " sur " + str(NBR_EPOCH))

        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)

        # Accumulate outputs
        accum_pred_img = []
        accum_pred_pos = []
        accum_scale = []
        batch_count = 0

        for img, pos in pbar:
            img = img.to(DEVICE)
            pos = pos.to(DEVICE)
            B = img.size(0)
            # Forward pass only, no backward yet
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                pred_img, pred_pos, scale = model(img, pos)
            accum_pred_img.append(pred_img)
            accum_pred_pos.append(pred_pos)
            accum_scale.append(scale)
            batch_count += 1

            # When 12 batches are accumulated, compute loss and backward
            if batch_count == BATCH_COMBINED // BATCH_SIZE:
                big_pred_img = torch.cat(accum_pred_img, dim=0)
                big_pred_pos = torch.cat(accum_pred_pos, dim=0)
                # Pour le scale, on prend la moyenne
                big_scale = torch.stack(accum_scale).mean()
                big_B = big_pred_img.size(0)
                logits = (big_pred_img @ big_pred_pos.T) * big_scale
                targets = torch.arange(big_B, device=DEVICE)

                loss_i2p = criterion(logits, targets)
                loss_p2i = criterion(logits.T, targets)
                loss = (loss_i2p + loss_p2i) / 2

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                total_loss += loss.item()

                # Reset accumulators
                accum_pred_img = []
                accum_pred_pos = []
                accum_scale = []
                batch_count = 0

        # Si il reste des batches non traités à la fin de l'epoch
        if batch_count > 0:
            optimizer.zero_grad()
            big_pred_img = torch.cat(accum_pred_img, dim=0)
            big_pred_pos = torch.cat(accum_pred_pos, dim=0)
            big_scale = torch.stack(accum_scale).mean()
            big_B = big_pred_img.size(0)
            logits = (big_pred_img @ big_pred_pos.T) * big_scale
            targets = torch.arange(big_B, device=DEVICE)

            loss_i2p = criterion(logits, targets)
            loss_p2i = criterion(logits.T, targets)
            loss = (loss_i2p + loss_p2i) / 2

            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()

        print(f"Fin epoch {epoch} loss tr moyenne {total_loss/max(1, (len(train_loader)//(BATCH_COMBINED // BATCH_SIZE)))}")
        print("Début de la validation : ")
        loss = model_validation(model, val_loader, criterion)
        scheduler.step(loss)
        print(f"Validation terminée, loss : {loss}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")


