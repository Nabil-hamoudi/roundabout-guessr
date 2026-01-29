import os
import zipfile
import gdown

DATASETS = {
    "paris_50k": {
        "id": "1Ht602iXoHgHuJ9hNJh9biDCdQwxGDzPH",  
        "output": "dataset_paris_50k.zip",
        "extract_path": "./datasets/paris"
    },
    "france_300k": {
        "id": "1PQ7r9Ijj5XKESN2vsECv_YgFcfE7xzAh",  
        "output": "dataset_france_300k.zip",
        "extract_path": "./datasets/france"
    }
}

MODELS = {
    "paris_50k" : {
        "id" : "qeKgNDOw9BItXGN3eOlYzIB1_",
        "output" : "model_paris_50k.pt"
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

if __name__ == "__main__":
    for dataset_name, info in DATASETS.items():
        zip_path = download_from_gdrive(info["id"], info["output"])
        extract_zip(zip_path, info["extract_path"], delete_zip=True)