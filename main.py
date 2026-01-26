import argparse
import sys
from pathlib import Path
from src import train_clip

def main():
    parser = argparse.ArgumentParser(
        description="CLI pour entraîner le modèle CLIP"
    )
    
    # Arguments positionnels (obligatoires)
    parser.add_argument(
        "nbr_epoch",
        type=int,
        help="Nombre d'epochs pour l'entraînement"
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="Taille du batch"
    )
    parser.add_argument(
        "dataset_folder",
        type=str,
        help="Chemin vers le dossier du dataset"
    )
    
    # Argument OPTIONNEL (commence par des tirets)
    parser.add_argument(
        '-bc', '--batchcombined',
        dest="batch_combined", # Nom utilisé dans args.batch_combined
        type=int,
        default=None,
        help="Taille du batch combiné"
    )

    
    args = parser.parse_args()

    json_path = Path(args.dataset_folder).joinpath("coordinates.json")
    images_path = Path(args.dataset_folder).joinpath("data")

    if not images_path.exists():
        print(f"Erreur : Le dossier {images_path} n'existe pas.")
        return

    if not json_path.exists():
        print(f"Erreur : Le fichier {json_path} n'existe pas.")
        return

    print(f"Dataset : {Path(args.dataset_folder)}")
    print(f"Batch combiné : {args.batch_combined}")

    # Lancer l'entraînement
    train_clip.train_clip(
        nbr_epoch=args.nbr_epoch,
        batch_size=args.batch_size,
        batch_combined=args.batch_combined,
        datajson=str(json_path),
        dataimages=str(images_path)
    )

if __name__ == "__main__":
    main()