import argparse
import sys
from pathlib import Path
from src import train_clip, embed_database

DESCRIPTIONCLI = "CLI pour ensemble des applications"
HELPSUBPARSER = 'Entrainement et lancement du modele'
HELPTRAIN = "Entraîner le modèle CLIP"
HELPEPOCH = "Nombre d'epochs pour l'entraînement"
HELPBATCH = "Taille du batch"
HELPBATCHCOMBI = "Taille du batch combiné"
HELPDATASET = "Chemin vers le dossier du dataset contenant un dossier data images et coordinates.json"
HELPMODEL = "Chemin vers le fichier du model"
HELPEMBEDING = "Chemin vers le fichier des embeddings"
HELPIMAGE = "Chemin vers le fichier de l'image"
HELPINITEMBEDDING = "Generer les embeddings"
HELPUSEMODEL = "Utiliser le modèle pour une image et les embeddings"
HELPTSNE = "Visualiser le tsne des embeddings"
HELPPCA = "Visualiser le pca des embeddings"
HELPPCAGEO = "Visualiser le pca des embeddings avec les coordonnées géographiques"

def main():
    parser = argparse.ArgumentParser(
        description="CLI pour ensemble des applications"
    )
    
    subparsers = parser.add_subparsers(help=HELPSUBPARSER, required=True, dest='subcommand')

####################################################################################

    trainparser = subparsers.add_parser("train", help=HELPTRAIN)
    # Arguments positionnels (obligatoires)
    trainparser.add_argument(
        "nbr_epoch",
        type=int,
        help=HELPEPOCH
    )
    trainparser.add_argument(
        "batch_size",
        type=int,
        help=HELPBATCH
    )
    trainparser.add_argument(
        "dataset_folder",
        type=str,
        help=HELPDATASET
    )
    
    # Argument OPTIONNEL (commence par des tirets)
    trainparser.add_argument(
        '-bc', '--batchcombined',
        dest="batch_combined", # Nom utilisé dans args.batch_combined
        type=int,
        default=None,
        help=HELPBATCHCOMBI
    )

####################################################################################
####################################################################################
    createembededparser = subparsers.add_parser("embedgen", help=HELPINITEMBEDDING)

    # Arguments positionnels (obligatoires)
    createembededparser.add_argument(
        "model_file",
        type=str,
        help=HELPMODEL
    )

    createembededparser.add_argument(
        "dataset_folder",
        type=str,
        help=HELPDATASET
    )

####################################################################################
####################################################################################
    usemodel = subparsers.add_parser("model", help=HELPUSEMODEL)

    # Arguments positionnels (obligatoires)
    usemodel.add_argument(
        "model_file",
        type=str,
        help=HELPMODEL
    )

    usemodel.add_argument(
        "embed_file",
        type=str,
        help=HELPEMBEDING
    )

    usemodel.add_argument(
        "image",
        type=str,
        help=HELPIMAGE
    )
####################################################################################
####################################################################################
    pca = subparsers.add_parser("pca", help=HELPPCA)

    pca.add_argument(
        "embed_file",
        type=str,
        help=HELPEMBEDING
    )
####################################################################################
####################################################################################
    pcageo = subparsers.add_parser("pca_geo", help=HELPPCAGEO)

    # Arguments positionnels (obligatoires)
    pcageo.add_argument(
        "dataset_folder",
        type=str,
        help=HELPDATASET
    )

    pcageo.add_argument(
        "embed_file",
        type=str,
        help=HELPEMBEDING
    )
####################################################################################
####################################################################################
    tsne = subparsers.add_parser("tsne", help=HELPTSNE)

    # Arguments positionnels (obligatoires)
    tsne.add_argument(
        "dataset_folder",
        type=str,
        help=HELPDATASET
    )

    tsne.add_argument(
        "embed_file",
        type=str,
        help=HELPEMBEDING
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

        if args.batch_combined is None:
            batch_combined = args.batch_size
        else:
            batch_combined = args.batch_combined

        # Lancer l'entraînement
        train_clip.train_clip(
            nbr_epoch=args.nbr_epoch,
            batch_size=args.batch_size,
            batch_combined=batch_combined,
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

        print(f"Embeddings : {embed}")
        print(f"Model : {model}")
        print(f"Image : {image}")

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
        print(f"Embeddings : {embed}")

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
        print(f"Embeddings : {embed}")

        embed_database.pca_geo(
            embed,
            json_path
        )

    elif args.subcommand in "pca":
        embed = Path(args.embed_file)
        if not embed.exists():
            print(f"Erreur : Le fichier {embed} n'existe pas.")
            return

        print(f"Embeddings : {embed}")

        embed_database.pca(
            embed
        )


if __name__ == "__main__":
    main()