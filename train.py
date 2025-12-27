import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import *
from model import *
from embed_database import *
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Chargement du JSON rond pts")
pos = get_roundabouts_pos("roundabouts.json")
print("Chargement des images")
imgs = load_images(len(pos))

#on doit avoir un ensemble de val

# si y'en a qu'un on peut rien faire en vrai
eligible_ids = [i for i, rb_imgs in enumerate(imgs) if len(rb_imgs) > 1]

#on en prend 40
val_indices = random.sample(eligible_ids, 40)

val_imgs = []
for idx in val_indices:
    img_for_val = imgs[idx].pop(0)
    val_imgs.append((idx, img_for_val))


def model_validation(model, train_imgs, val_imgs):
    model.eval()
    
    db_train = create_database(train_imgs, model) 
    
    correct = 0
    total = 0
    pbar = tqdm(val_imgs, desc=f"Validation", unit="query", leave=False)

    with torch.no_grad():
        for true_id, img in pbar:
            predicted_id = get_roundabout(db_train, img, model)
            
            if predicted_id == true_id:
                correct += 1
            total += 1
            pbar.set_postfix({"accuracy": f"{correct/total:.4f}"})
    accuracy = (correct / total) * 100

    return accuracy

dataset = RoundAboutTrainingDataset(imgs, pos)

model = BaseEmbed().to(DEVICE)

#Gérer la validation après déjà je veux faire en sorte que ça forward
train_loader = DataLoader(dataset, batch_size=8)

criterion = nn.TripletMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1,11):
    print("Début de l'epoch " + str(epoch) + " sur 10")

    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)

    for img_a, img_b, img_c in pbar:
        img_a, img_b, img_c = img_a.to(DEVICE), img_b.to(DEVICE), img_c.to(DEVICE)
        optimizer.zero_grad()
        #img a = l'ancre, b = l'image actuelle, c = négative
        pred_a = model(img_a)
        pred_b = model(img_b)
        pred_c = model(img_c)

        loss = criterion(pred_a, pred_b, pred_c)

        loss.backward()
        optimizer.step()
        #print(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        total_loss += loss.item()
    
    print(f"Fin epoch {epoch} loss tr moyenne {total_loss/len(train_loader)}")
    print("Début de la validation : ")
    acc = model_validation(model, imgs, val_imgs)
    print(f"Validation terminée, accuracy : {acc}")

    
