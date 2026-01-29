import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.cross.dataset_cross import *
from src.cross.model_cross import *
from src.embed_database import *
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime
import numpy as np
import random
from collections import defaultdict

from src.training_utils import criterion_duplicates, panorama_split
from src.validation_utils import model_validation, validate_retrieval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

NB_EPOCH = 150
BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 32
BATCH_COMBINED = 600
DATAFOLDER = "gen_fr"
DATAJSON = DATAFOLDER + "/coordinates_paris.json"
DATAIMAGES = DATAFOLDER + "/data_paris"
DATASAT = DATAFOLDER + "/sat_paris"

LAT_MIN_F, LAT_MAX_F = 41.3, 51.1
LON_MIN_F, LON_MAX_F = -5.1, 9.6

def train_cross(nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE, batch_combined=BATCH_COMBINED, data_json=DATAJSON, data_images=DATAIMAGES, data_sat=DATASAT, want_france=False):
    NB_EPOCH = nb_epoch
    BATCH_SIZE = batch_size
    BATCH_COMBINED = batch_combined

    scaler = torch.amp.GradScaler(DEVICE_STR)
    print("Chargement des images")
    imgs = get_images_paths(data_images)
    #print(len(imgs))

    print("Chargement du JSON rond pts")
    pos = get_images_pos(data_json)
    log_history = []

    sat_paths = get_images_paths(data_sat)
    #print(len(sat_paths), data_sat)

    train_dataset, val_dataset, train_dataset_clean = panorama_split(imgs, pos, sat_paths, want_clean=True)


    if want_france:
        model = CrossEncoder(LAT_MIN=LAT_MIN_F, LAT_MAX=LAT_MAX_F, LON_MIN=LON_MIN_F, LON_MAX=LON_MAX_F).to(DEVICE)
    else:
        model = CrossEncoder().to(DEVICE)

    #Gérer la validation après déjà je veux faire en sorte que ça forward
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    train_clean_loader = DataLoader(train_dataset_clean, batch_size=INFERENCE_BATCH_SIZE, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    optimizer = torch.optim.AdamW([
        {'params': model.image_encoder.parameters(), 'lr': 1e-4}, 
        {'params': model.sat_encoder.parameters(), 'lr': 1e-4}, 
        {'params': model.location_encoder.parameters(), 'lr': 1e-4}, 
        {'params': [model.logit_scale], 'lr': 1e-3} 
    ], weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-5)  # Descend sur 30 epochs

    def get_accumulated_loss(accum_pred_img, accum_pred_pos, accum_pred_sat, accum_scale, accum_pos_coords):
        big_pred_img = torch.cat(accum_pred_img, dim=0)
        big_pred_pos = torch.cat(accum_pred_pos, dim=0)
        big_pos_coords = torch.cat(accum_pos_coords, dim=0)
        big_pred_sat = torch.cat(accum_pred_sat, dim=0)
        # Pour le scale, on prend la moyenne
        big_scale = torch.stack(accum_scale).mean()

        loss = criterion_duplicates(
            big_pred_img, 
            big_pred_pos, 
            big_scale, 
            big_pos_coords,
            big_pred_sat,
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

        accum_pred_img = []; accum_pred_pos = []; accum_scale = []; accum_pos_coords = []; accum_pred_sat = []
        batch_count = 0

        for img, pos, sat in pbar:
            img = img.to(DEVICE)
            pos = pos.to(DEVICE)
            sat = sat.to(DEVICE)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                pred_img, pred_pos, pred_sat, scale = model(img, pos, sat)
            accum_pred_img.append(pred_img); accum_pred_pos.append(pred_pos); accum_pred_sat.append(pred_sat); accum_pos_coords.append(pos); accum_scale.append(scale)
            batch_count += 1

            if batch_count == BATCH_COMBINED // BATCH_SIZE:
                loss = get_accumulated_loss(accum_pred_img, accum_pred_pos, accum_pred_sat, accum_scale, accum_pos_coords)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                total_loss += loss.item()

                accum_pred_img = []; accum_pred_pos = []; accum_scale = []; accum_pos_coords = []; accum_pred_sat = []
                batch_count = 0
            break

        if batch_count > 0:
            loss = get_accumulated_loss(accum_pred_img, accum_pred_pos, accum_pred_sat, accum_scale, accum_pos_coords)
            optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            total_loss += loss.item()

        print(f"Fin epoch {epoch} loss tr moyenne {total_loss/max(1, (len(train_loader)/(BATCH_COMBINED / BATCH_SIZE)))}")
        #print(f"logit_scale = {model.logit_scale.exp().item():.2f}")

        print("Début de la validation : ")
        #loss = model_validation(model, val_loader)#, criterion)
        loss = model_validation(model, val_loader, has_sat=True)#, criterion)

        if epoch % 5 == 0 or epoch == 1:
            loss = validate_retrieval(model, train_clean_loader, val_loader)
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


if __name__ == "__main__":
    train_cross()