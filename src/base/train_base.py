import torch
import torch.nn as nn
from tqdm import tqdm
from src.base.dataset import *
from src.base.model import *
from src.embed_database import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime
from src.training_utils import *
from src.validation_utils import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

NB_EPOCH = 101
BATCH_SIZE = 32
BATCH_COMBINED = 1200
DATA_JSON = "yo/coordinates.json"
DATA_IMAGES = "yo/data"

LAT_MIN_F, LAT_MAX_F = 41.3, 51.1
LON_MIN_F, LON_MAX_F = -5.1, 9.6

def train_base(nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE, batch_combined=BATCH_COMBINED, data_json=DATA_JSON, data_images=DATA_IMAGES, want_france=False):
    NB_EPOCH = nb_epoch
    BATCH_SIZE = batch_size
    BATCH_COMBINED = batch_combined
    DATA_JSON = data_json
    DATA_IMAGES = data_images

    scaler = torch.amp.GradScaler(DEVICE_STR)
    print("Chargement du JSON des coordonnées")
    pos = get_images_pos(DATA_JSON)
    print("Chargement des images")
    imgs = get_images_paths(DATA_IMAGES)
    log_history = []

    train_dataset, val_dataset = geographic_split(imgs, pos)

    print(train_dataset.is_train)
    print(val_dataset.is_train)

    if want_france:
        model = MixedEncoder(LAT_MIN=LAT_MIN_F, LAT_MAX=LAT_MAX_F, LON_MIN=LON_MIN_F, LON_MAX=LON_MAX_F).to(DEVICE)
    else:
        model = MixedEncoder().to(DEVICE)
    #model.load_state_dict(torch.load("model_epoch_3.pt"))

    #Gérer la validation après déjà je veux faire en sorte que ça forward
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)  # Descend sur 30 epochs

    def get_accumulated_loss(accum_pred_img, accum_pred_pos, accum_scale, accum_pos_coords):
        big_pred_img = torch.cat(accum_pred_img, dim=0)
        big_pred_pos = torch.cat(accum_pred_pos, dim=0)
        big_pos_coords = torch.cat(accum_pos_coords, dim=0)
        
        big_scale = torch.stack(accum_scale).mean()
        
        loss = criterion_duplicates(
            big_pred_img, 
            big_pred_pos, 
            big_scale, 
            big_pos_coords,
        )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        return loss

    for epoch in range(1, NB_EPOCH):
        print("Début de l'epoch " + str(epoch) + " sur " + str(NB_EPOCH))

        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)

        accum_pred_img = []; accum_pred_pos = []; accum_scale = []; accum_pos_coords = []
        batch_count = 0

        for img, pos in pbar:
            img = img.to(DEVICE)
            pos = pos.to(DEVICE)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                pred_img, pred_pos, scale = model(img, pos)
            accum_pred_img.append(pred_img); accum_pred_pos.append(pred_pos); accum_pos_coords.append(pos); accum_scale.append(scale)
            batch_count += 1

            if batch_count == BATCH_COMBINED // BATCH_SIZE:
                loss = get_accumulated_loss(accum_pred_img, accum_pred_pos, accum_scale, accum_pos_coords)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                total_loss += loss.item()

                accum_pred_img = []; accum_pred_pos = []; accum_scale = []; accum_pos_coords = []
                batch_count = 0
            break

        if batch_count > 0:
            loss = get_accumulated_loss(accum_pred_img, accum_pred_pos, accum_scale, accum_pos_coords)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()

        print(f"Fin epoch {epoch} loss tr moyenne {total_loss/max(1, (len(train_loader)/(BATCH_COMBINED / BATCH_SIZE)))}")
        #print(f"logit_scale = {model.logit_scale.exp().item():.2f}")

        print("Début de la validation : ")
        loss = model_validation(model, val_loader)#, criterion)

        log_entry = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)/(BATCH_COMBINED/BATCH_SIZE)),
            "lr": optimizer.param_groups[0]['lr'],
            "logit_scale": model.logit_scale.exp().item(),
            "median_error_km": loss,
            "timestamp": datetime.now().isoformat()
        }
        log_history.append(log_entry)
        with open("training_log.json", "w") as f:
            json.dump(log_history, f, indent=2)
        scheduler.step()
        print(f"Validation terminée, loss : {loss}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")


