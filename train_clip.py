import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from dataset import *
from model2 import *
from embed_database import *
import random
from torch.optim.lr_scheduler import LinearLR
from loss import *
import copy



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NBR_EPOCH = 101

BATCH_SIZE = 32
BATCH_COMBINED = 300

def model_validation(model, val_imgs, criterion):
    model.eval()
    
    total_loss = 0
    pbar = tqdm(val_imgs, desc=f"Validation", unit="query", leave=False)

    with torch.no_grad():
        for img, pos in pbar:
            img = img.to(DEVICE)
            pos = pos.to(DEVICE)
            if img.size(0) < 2:
                continue 
            B = img.size(0)
            logits, labels = model(img, pos)
            
            loss_i = criterion(logits, labels)      # Image -> Loc
            loss_l = criterion(logits.t(), labels)  # Loc -> Image
            loss = (loss_i + loss_l) / 2

            #print(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()

    return total_loss/len(val_imgs)

#gentiment généré par gemini
#l'implémentation maison n'étant pas suffisamment efficace...
class MoCoWrapper(nn.Module):
    def __init__(self, base_encoder, queue_size=4096):
        super().__init__()
        self.base_encoder = base_encoder
        self.queue_size = queue_size
        
        # On stocke les coordonnées GPS brutes (pas les embeddings !)
        # Shape: (2, 4096)
        self.register_buffer("gps_queue", torch.randn(2, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Ajoute les GPS du batch actuel dans la file d'attente """
        batch_size = gps.shape[0]
        ptr = int(self.queue_ptr)
        
        # Si le batch est plus petit que l'espace restant (cas fin d'epoch)
        if ptr + batch_size > self.queue_size:
            ptr = 0 # On repart au début (simplification pour éviter les bugs d'index)
            
        self.gps_queue[:, ptr:ptr + batch_size] = gps.t()
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, img, loc):
        # 1. Image Embedding (Lourd -> fait 1 seule fois)
        # base_encoder retourne (img_emb, loc_emb, scale)
        # On n'utilise que img_emb ici
        img_features = self.base_encoder.image_encoder(img)
        img_features = F.normalize(img_features, dim=1)
        
        # Récupération du scale (température)
        logit_scale = self.base_encoder.logit_scale.exp()

        if self.training:
            # --- MODE TRAIN : AVEC QUEUE ---
            
            # A. Embedding du batch de location actuel
            loc_features = self.base_encoder.location_encoder(loc)
            loc_features = F.normalize(loc_features, dim=1)
            
            # B. Embedding de la Queue (Recalculé à la volée !)
            # C'est ultra-rapide (quelques millisecondes)
            with torch.no_grad():
                queue_gps = self.gps_queue.t() # (K, 2)
                queue_feat = self.base_encoder.location_encoder(queue_gps)
                queue_feat = F.normalize(queue_feat, dim=1)
            
            # C. Concaténation : [Batch (Positifs) | Queue (Négatifs)]
            # (Batch + K, Dim)
            all_loc_features = torch.cat([loc_features, queue_feat], dim=0)
            
            # D. Calcul de similarité
            # (B, Dim) @ (B + K, Dim).T -> (B, B + K)
            logits = logit_scale * (img_features @ all_loc_features.t())
            
            # E. Labels
            # La bonne réponse pour l'image 0 est à l'index 0
            # La bonne réponse pour l'image 1 est à l'index 1...
            labels = torch.arange(img.size(0), device=img.device)
            
            # F. Mise à jour de la queue
            self.dequeue_and_enqueue(loc)
            
            return logits, labels
            
        else:
            # --- MODE VAL : SANS QUEUE (Batch vs Batch) ---
            loc_features = self.base_encoder.location_encoder(loc)
            loc_features = F.normalize(loc_features, dim=1)
            
            # Matrice carrée (B, B)
            logits = logit_scale * (img_features @ loc_features.t())
            labels = torch.arange(img.size(0), device=img.device)
            
            return logits, labels
        
import copy
if __name__ == "__main__":

    print("Chargement du JSON rond pts")
    pos = get_images_pos("yo/coordinates.json")
    print("Chargement des images")
    imgs = get_images_paths()


    dataset = ImagesPosDataset(imgs, pos)
    generator1 = torch.Generator().manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(dataset)), 
        [0.99, 0.01], 
        generator=generator1
    )

    train_dataset = Subset(dataset, train_indices.indices)
    val_dataset = Subset(copy.deepcopy(dataset), val_indices.indices)

    base_model = MixedEncoder().to(DEVICE)
    model = MoCoWrapper(base_model, queue_size=4096).to(DEVICE)
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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)

        # Accumulate outputs
        accum_pred_img = []
        accum_pred_pos = []
        batch_count = 0

        for img, pos in pbar:
            img, pos = img.to(DEVICE), pos.to(DEVICE)

            if img.size(0) < BATCH_SIZE:
                continue
            
            optimizer.zero_grad()
            
            logits, labels = model(img, pos)
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            #break

        print(f"Fin epoch {epoch} loss tr moyenne {total_loss/len(train_loader)}")
        print("Début de la validation : ")
        loss = model_validation(model, val_loader, criterion)
        scheduler.step(loss)
        print(f"Validation terminée, loss : {loss}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

