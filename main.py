import argparse
import sys
from pathlib import Path
from src import train_clip, embed_database

def main():
    parser = argparse.ArgumentParser(
        description="CLI pour ensemble des applications"
    )
    
    subparsers = parser.add_subparsers(help='Entrainement et lancement du modele', required=True, dest='subcommand')

####################################################################################

    trainparser = subparsers.add_parser("train", help="Entraîner le modèle CLIP")
    # Arguments positionnels (obligatoires)
    trainparser.add_argument(
        "nbr_epoch",
        type=int,
        help="Nombre d'epochs pour l'entraînement"
    )
    trainparser.add_argument(
        "batch_size",
        type=int,
        help="Taille du batch"
    )
    trainparser.add_argument(
        "dataset_folder",
        type=str,
        help="Chemin vers le dossier du dataset"
    )
    
    # Argument OPTIONNEL (commence par des tirets)
    trainparser.add_argument(
        '-bc', '--batchcombined',
        dest="batch_combined", # Nom utilisé dans args.batch_combined
        type=int,
        default=None,
        help="Taille du batch combiné"
    )

####################################################################################
####################################################################################
    createembededparser = subparsers.add_parser("embedgen", help="Generer les embeddings")

    # Arguments positionnels (obligatoires)
    createembededparser.add_argument(
        "model_file",
        type=str,
        help="Chemin vers le fichier du model"
    )

    createembededparser.add_argument(
        "dataset_folder",
        type=str,
        help="Chemin vers le dossier du dataset"
    )

####################################################################################
####################################################################################
    usemodel = subparsers.add_parser("model", help="utiliser model")

    # Arguments positionnels (obligatoires)
    usemodel.add_argument(
        "model_file",
        type=str,
        help="Chemin vers le fichier du model"
    )

    usemodel.add_argument(
        "embed_file",
        type=str,
        help="Chemin vers les embeddings"
    )

    usemodel.add_argument(
        "image",
        type=str,
        help="Chemin vers l'image'"
    )
####################################################################################
####################################################################################
    pca = subparsers.add_parser("pca", help="visualise le pca")

    pca.add_argument(
        "embed_file",
        type=str,
        help="Chemin vers les embeddings"
    )
####################################################################################
####################################################################################
    pcageo = subparsers.add_parser("pca_geo", help="visualise le pca geo")

    # Arguments positionnels (obligatoires)
    pcageo.add_argument(
        "dataset_folder",
        type=str,
        help="Chemin vers le dossier du dataset"
    )

    pcageo.add_argument(
        "embed_file",
        type=str,
        help="Chemin vers les embeddings"
    )
####################################################################################
####################################################################################
    tsne = subparsers.add_parser("tsne", help="visualise le tsne")

    # Arguments positionnels (obligatoires)
    tsne.add_argument(
        "dataset_folder",
        type=str,
        help="Chemin vers le dossier du dataset"
    )

    tsne.add_argument(
        "embed_file",
        type=str,
        help="Chemin vers les embeddings"
    )
####################################################################################


    args = parser.parse_args()

    print(str(args))

    if args.subcommand == "train":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        images_path = Path(args.dataset_folder).joinpath("data")

        if not images_path.exists():
            print(f"Erreur : Le dossier {images_path} n'existe pas.")
            return

        if not json_path.exists():
            print(f"Erreur : Le fichier {json_path} n'existe pas.")
            return

        print(f"Dataset : {Path(args.dataset_folder)}")
        print(f"Batch combined : {args.batch_combined}")

        # Lancer l'entraînement
        train_clip.train_clip(
            nbr_epoch=args.nbr_epoch,
            batch_size=args.batch_size,
            batch_combined=args.batch_combined,
            datajson=str(json_path),
            dataimages=str(images_path)
        )

    elif args.subcommand == "embedgen":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        images_path = Path(args.dataset_folder).joinpath("data")
        model = Path(args.model_file)
        if not images_path.exists():
            print(f"Erreur : Le dossier {images_path} n'existe pas.")
            return

        if not json_path.exists():
            print(f"Erreur : Le fichier {json_path} n'existe pas.")
            return

        if not model.exists():
            print(f"Erreur : Le fichier {model} n'existe pas.")
            return

        print(f"Dataset : {Path(args.dataset_folder)}")
        print(f"Model : {model}")

        embed_database.init(
            model,
            images_path,
            json_path
        )


    elif args.subcommand == "model":
        embed = Path(args.embed_file)
        model = Path(args.model_file)
        image = Path(args.image)
    
        if not image.exists():
            print(f"Erreur : Le fichier {image} n'existe pas.")
            return

        if not embed.exists():
            print(f"Erreur : Le fichier {json_path} n'existe pas.")
            return

        if not model.exists():
            print(f"Erreur : Le fichier {model} n'existe pas.")
            return

        print(f"embedings : {embed}")
        print(f"Model : {model}")
        print(f"image : {image}")

        embed_database.usemodel(
            model,
            embed,
            image
        )

    elif args.subcommand in "tsne":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        embed = Path(args.embed_file)
        if not embed.exists():
            print(f"Erreur : Le fichier {embed} n'existe pas.")
            return

        if not json_path.exists():
            print(f"Erreur : Le fichier {json_path} n'existe pas.")
            return

        print(f"Dataset : {Path(args.dataset_folder)}")
        print(f"embedings : {embed}")

        embed_database.tsne(
            embed,
            json_path
        )

    elif args.subcommand in "pca_geo":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        embed = Path(args.embed_file)
        if not embed.exists():
            print(f"Erreur : Le fichier {embed} n'existe pas.")
            return

        if not json_path.exists():
            print(f"Erreur : Le fichier {json_path} n'existe pas.")
            return

        print(f"Dataset : {Path(args.dataset_folder)}")
        print(f"embedings : {embed}")

        embed_database.pca_geo(
            embed,
            json_path
        )

    elif args.subcommand in "pca":
        embed = Path(args.embed_file)
        if not embed.exists():
            print(f"Erreur : Le fichier {embed} n'existe pas.")
            return

        print(f"Embedings : {embed}")

        embed_database.pca(
            embed
        )


if __name__ == "__main__":
    main()