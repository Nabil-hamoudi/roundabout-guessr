import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.dataset import *
from tqdm import tqdm
import cv2
from src.model import MixedEncoder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
def create_database(imgs, pos, model):

    print("Creating the databse")
    model.eval()
    """
    Le format là en gros c'est 
        roundabout : [liste des vecteurs d'embed]

    -> C'est super long, on verra plus tard pour opti.
    -> ça a l'avantage d'être super simple !
    """
    ids = []
    elems = []

    dataset = ImagesPosDataset(imgs, pos, want_index=True)
    loader = DataLoader(dataset, batch_size=64)

    pbar = tqdm(loader, desc=f"Calcul des embeds", unit="batch", leave=False)
    with torch.no_grad():
        for _, pos, i_batch in pbar:
            pos = pos.to(DEVICE)
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                vecs = model.location_encoder(pos).detach().cpu().numpy()

            for idx,vec in zip(i_batch, vecs):
                i = idx.item()

                ids.append(i)
                elems.append(vec)
    a = {
        "ids" : ids,
        "elems" : elems
    }
    torch.save(a, 'embeddings_db.pt')

    #print(elems)
    return a

def load_database(path='embeddings_db.pt'):
    return torch.load(path, weights_only=False)

def get_closest(db, img, model, k = 5):
    model.eval()
    
    img_tensor = compat_transform(image=img)["image"]
    img_tensor = img_tensor.to(DEVICE).unsqueeze(0) 

    with torch.no_grad():
        query_vec = model.image_encoder(img_tensor)
        query_vec = F.normalize(query_vec, p=2, dim=1)

    db_vecs = torch.tensor(np.array(db['elems']), device=DEVICE)
    db_ids = np.array(db['ids'])

    #query x dbvect.T
    scores = torch.mm(query_vec, db_vecs.T)
    
    best_scores, best_indices = torch.topk(scores, k=k, dim=1)
    
    best_indices = best_indices.cpu().numpy()[0]
    best_scores = best_scores.cpu().numpy()[0]
    
    results = []
    for rank, idx in enumerate(best_indices):
        result_id = db_ids[idx]
        score = best_scores[rank]
        results.append((result_id, score))
        
    return results

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_pca(embeddings_path='embeddings_db.pt'):
    """Visualisation PCA 2D simple"""
    
    # Charger
    data = torch.load(embeddings_path, weights_only=False)
    if isinstance(data, dict):
        embeddings = np.array(data['elems'])
        ids = np.array(data['ids'])
    else:
        embeddings = data.elems
        ids = data.ids
    
    print(f"PCA sur {len(embeddings)} embeddings de dim {embeddings.shape[1]}")
    
    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=ids, cmap='tab20', 
                alpha=0.6, s=20, edgecolors='k', linewidth=0.2)
    
    var1, var2 = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({var1:.1%})')
    plt.ylabel(f'PC2 ({var2:.1%})')
    plt.title(f'PCA - Variance totale: {var1+var2:.1%}')
    plt.colorbar(label='ID')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_pca_geo(embeddings_path='embeddings_db.pt', pos_dict=None):
    """Compare PCA et positions géographiques"""
    
    # Charger embeddings
    data = torch.load(embeddings_path, weights_only=False)
    if isinstance(data, dict):
        embeddings = np.array(data['elems'])
        ids = np.array(data['ids'])
    else:
        embeddings = data.elems
        ids = data.ids
    
    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)
    
    # Extraire positions géo
    geo_coords = []
    pca_valid = []
    for i, id_val in enumerate(ids):
        geo_coords.append(pos_dict[id_val])
        pca_valid.append(pca_coords[i])

    geo_coords = np.array(geo_coords)
    pca_valid = np.array(pca_valid)
    
    # Plot côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA
    sc1 = ax1.scatter(pca_valid[:, 0], pca_valid[:, 1], 
                     c=geo_coords[:, 0], cmap='RdYlBu_r', s=30, edgecolors='k', linewidth=0.3)
    var1, var2 = pca.explained_variance_ratio_
    ax1.set_xlabel(f'PC1 ({var1:.1%})')
    ax1.set_ylabel(f'PC2 ({var2:.1%})')
    ax1.set_title('Espace latent PCA')
    ax1.grid(alpha=0.3)
    plt.colorbar(sc1, ax=ax1, label='Latitude')
    
    # Géographie
    print(len(geo_coords))
    sc2 = ax2.scatter(geo_coords[:, 1], geo_coords[:, 0], 
                     c=geo_coords[:, 0], cmap='RdYlBu_r', s=30, edgecolors='k', linewidth=0.3)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Positions géographiques')
    ax2.grid(alpha=0.3)
    plt.colorbar(sc2, ax=ax2, label='Latitude')
    
    plt.tight_layout()
    plt.show()
from sklearn.manifold import TSNE

def visualize_tsne(embeddings_path='embeddings_db.pt', pos_dict=None):
    """Visualisation t-SNE (bien meilleure pour les clusters)"""
    
    # Charger les données
    data = torch.load(embeddings_path, weights_only=False)
    if isinstance(data, dict):
        embeddings = np.array(data['elems'])
        ids = np.array(data['ids'])
    else:
        embeddings = data.elems
        ids = data.ids

    print(f"t-SNE sur {len(embeddings)} embeddings... (ça peut prendre un peu de temps)")
    
    # t-SNE : on réduit à 2 dimensions pour l'affichage
    # perplexity=30 est standard, à varier entre 5 et 50 selon la taille des clusters
    tsne = TSNE(n_components=2, perplexity=200, random_state=42, init="pca")
    coords = tsne.fit_transform(embeddings)
    
    # Récupérer les couleurs (Latitude par exemple si dispo, sinon ID)
    colors = ids
    label_name = "ID"
    
    if pos_dict is not None:
        # Si on a les positions, on colorie par Latitude pour voir le dégradé géographique
        geo_colors = []
        for id_val in ids:
            geo_colors.append(pos_dict[id_val][0]) # 0 pour Lat, 1 pour Long
        colors = geo_colors
        label_name = "Latitude"

    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='Spectral', 
                alpha=0.7, s=30, edgecolors='k', linewidth=0.1)
    
    plt.colorbar(scatter, label=label_name)
    plt.title(f't-SNE Visualization (Embed dim: {embeddings.shape[1]})')
    plt.axis('off') # Les axes t-SNE n'ont pas d'unité signifiante
    plt.tight_layout()
    plt.show()



def visualize_geo(pos_dict=None):
    geo_coords = []
    found_count = 0
    
    for id_val in range(len(pos_dict)):
        geo_coords.append(pos_dict[id_val])
        found_count += 1
            
    geo_coords = np.array(geo_coords)
    print(f"Affichage de {found_count} points géographiques.")

    # 3. Plot
    plt.figure(figsize=(10, 10)) # Format carré pour éviter d'écraser la France
    
    # x = Longitude (colonne 1), y = Latitude (colonne 0)
    # On garde la couleur par latitude (cmap='RdYlBu_r') pour le style
    sc = plt.scatter(geo_coords[:, 1], geo_coords[:, 0], 
                     c=geo_coords[:, 0], cmap='RdYlBu_r', 
                     s=30, edgecolors='k', linewidth=0.3, alpha=0.8)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Distribution Géographique du Dataset')
    plt.grid(alpha=0.3, linestyle='--')
    
    # Important pour une carte : ratio égal pour ne pas déformer les distances
    plt.axis('equal') 
    
    plt.colorbar(sc, label='Latitude')
    plt.tight_layout()
    plt.show()


def init(model_path, data_path, json_path):
    pos = get_images_pos(json_path)
    imgs = get_images_paths(data_path)

    model = MixedEncoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    db = create_database(imgs, pos, model)
    torch.save(db, "embeddings_db.pt")


def tsne(embeding_path, json_path):
    pos = get_images_pos(json_path)
    visualize_tsne(embeding_path, pos_dict=pos)


def pca(embeding_path):
    visualize_pca(embeding_path)

def pca_geo(embeding_path, json_path):
    pos = get_images_pos(json_path)
    compare_pca_geo(embeding_path, pos_dict=pos)

def usemodel(model_path, embeding_path, image_path):
    model = MixedEncoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    a = torch.load(embeding_path, weights_only=False)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(get_closest(a, img, model))
