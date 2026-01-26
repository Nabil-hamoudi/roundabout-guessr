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

    dataset_path = Path(args.dataset_folder).resolve()

    if not dataset_path.exists():
        print(f"Erreur : Le dossier {dataset_path} n'existe pas.")
        return

    print(f"Dataset : {dataset_path}")
    print(f"Batch combiné : {args.batch_combined}")

    # Lancer l'entraînement
    train_clip(
        nbr_epoch=args.nbr_epoch,
        batch_size=args.batch_size,
        batch_combined=args.batch_combined,
        datafolder=str(dataset_path),
    )

if __name__ == "__main__":
    main()