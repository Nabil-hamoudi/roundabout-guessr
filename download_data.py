import os
import zipfile
import gdown
import sys
import argparse

DATASETS = {
    "paris_1k" : {
        "id" : "1ulp6vD-rpDRm-rYo6CirefnI23k5ZImk",
        "output" : "dataset_paris_1k.zip",
        "extract_path" : "./datasets/paris_1k"
    },
    "paris_50k": {
        "id": "1Ht602iXoHgHuJ9hNJh9biDCdQwxGDzPH",  
        "output": "dataset_paris_50k.zip",
        "extract_path": "./datasets/paris_50k"
    },
    "paris_100k": {
        "id" : "1_J98Wfn-7yjhDlurKxnA5QkM0dTYcKUS",
        "output" : "dataset_paris_100k.zip",
        "extract_path": "./datasets/paris_100k"
    },
    "france_300k": {
        "id": "1PQ7r9Ijj5XKESN2vsECv_YgFcfE7xzAh",  
        "output": "dataset_france_300k.zip",
        "extract_path": "./datasets/france_300k"
    },
    "france_50k" : {
        "id" : "1VElOIWDLL83oL-OrIfO-i7G07vkbTfpn",
        "output" : "dataset_france_50k.zip",
        "extract_path": "./datasets/france_50k"
    }
}

MODELS = {
    "paris_50k" : {
        "id" : "1tcvvOx-qeKgNDOw9BItXGN3eOlYzIB1_",
        "output" : "model_paris_50k.pt"
    }
}

EMBEDDINGS = {
    "paris_50k" : {
        "id": "10chFS8j1vi3LKs4luc_EJiBSthro-jtt",
        "output": "embeddings_paris_50k.pt"
    }
}

COORDINATES = {
    "paris_50k" : {
        "id": "1T3rbsU2nFRjj8hwzahFcEevR6vVNJ-EG",
        "output": "coordinates_paris_50k.json"
    },
    "paris_100k" :{
        "id" : "12mlWetNmBiD1pRtkQ00hyQQzin9zLub1",
        "output" : "coordinates_paris_100k.json"
    }

}


def download_from_gdrive(file_id, output_filename):
    """
    Télécharge un fichier depuis Google Drive via son ID.
    Renvoie le chemin du fichier téléchargé.
    """
    if os.path.exists(output_filename):
        print(f"Fichier déjà présent : {output_filename}")
        return output_filename
        
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Téléchargement de {output_filename}...")
    
    gdown.download(url, output_filename, quiet=False)
    
    return output_filename

def extract_zip(zip_path, extract_path, delete_zip=False):
    """
    Décompresse un zip dans un dossier donné.
    Option: supprimer le zip après coup pour gagner de la place.
    """
    if not os.path.exists(zip_path):
        print(f"Erreur : Le fichier {zip_path} n'existe pas.")
        return

    # Création du dossier cible s'il n'existe pas
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

    print(f"Extraction de {zip_path} vers {extract_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extraction terminée.")
        
        if delete_zip:
            os.remove(zip_path)
            print(f"Archive zip supprimée.")
            
    except zipfile.BadZipFile:
        print(f"Erreur : Ce n'est pas un zip valide.")
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")

def list_resources():
    print("\n --- RESSOURCES DISPONIBLES --- \n")
    
    print("DATASETS :")
    for key in DATASETS:
        print(f"  - {key}")
        
    print("\nMODELES :")
    for key in MODELS:
        print(f"  - {key}")

    print("\nEMBEDDINGS :")
    for key in EMBEDDINGS:
        print(f"  - {key}")

    print("\nCOORDONNEES :")
    for key in COORDINATES:
        print(f"  - {key}")
    print()

def process_download(name, resource_type, keep_zip):
    target_dict = None
    if resource_type == 'dataset':
        target_dict = DATASETS
    elif resource_type == 'model':
        target_dict = MODELS
    elif resource_type == 'embedding':
        target_dict = EMBEDDINGS
    elif resource_type == 'coordinates':
        target_dict = COORDINATES

    if name not in target_dict:
        print(f"Erreur : {name} introuvable dans la catégorie '{resource_type}'.")
        return

    conf = target_dict[name]
    print(f"🚀 Traitement de {resource_type.upper()} : {name}")

    file_path = download_from_gdrive(conf["id"], conf["output"])
    
    if file_path and resource_type == 'dataset':
        extract_zip(file_path, conf["extract_path"], delete_zip=not keep_zip)
    elif file_path:
        print(f"✅ Fichier prêt : {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestionnaire de téléchargement GDrive")
    subparsers = parser.add_subparsers(dest='command', help="Commandes")

    subparsers.add_parser('list', help="Lister les ressources")

    dl_parser = subparsers.add_parser('download', help="Télécharger une ressource")
    dl_parser.add_argument('name', type=str, help="Nom de la ressource (ex: paris_50k)")
    
    dl_parser.add_argument('--type', type=str, required=True, 
                           choices=['dataset', 'model', 'embedding', 'coordinates'],
                           help="Le type de ressource à télécharger")
    
    dl_parser.add_argument('--keep-zip', action='store_true', help="Garder le zip après extraction (datasets uniquement)")

    args = parser.parse_args()

    if args.command == 'list':
        list_resources()
    elif args.command == 'download':
        process_download(args.name, args.type, args.keep_zip)
    else:
        parser.print_help()