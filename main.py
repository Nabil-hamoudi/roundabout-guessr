import argparse
import sys
from pathlib import Path

import torch
DESCRIPTIONCLI = "CLI pour ensemble des applications"
HELPSUBPARSER = 'Entrainement et lancement du modele'
HELPTRAIN = "Entraîner le modèle"
HELPEPOCH = "Nombre d'epochs pour l'entraînement"
HELPBATCH = "Taille du batch"
HELPBATCHCOMBI = "Taille du batch combiné"
HELPDATASET = "Chemin vers le dossier du dataset contenant un dossier data images et coordinates.json"
HELPMODEL = "Chemin vers le fichier du model"
HELPEMBEDING = "Chemin vers le fichier des embeddings"
HELPIMAGE = "Chemin vers le fichier de l'image"
HELPINITEMBEDDING = "Generer les embeddings"
HELPUSEMODEL = "Récupère les points les plus proches d'une image donnée"
HELPTSNE = "Visualiser le tsne des embeddings"
HELPPCA = "Visualiser le pca des embeddings"
HELPPCAGEO = "Visualiser le pca des embeddings avec les coordonnées géographiques"
HELPCROSS = 'Utiliser le modèle cross-view'
HELPCOORDINATES = "Chemin vers le fichier des coordonnées"

def main():
    parser = argparse.ArgumentParser(
        description="CLI pour ensemble des applications"
    )
    
    subparsers = parser.add_subparsers(help=HELPSUBPARSER, required=True, dest='subcommand')

####################################################################################

    train_parser = subparsers.add_parser("train", help=HELPTRAIN)
    # Arguments positionnels (obligatoires)
    train_parser.add_argument(
        "nb_epoch",
        type=int,
        help=HELPEPOCH
    )
    train_parser.add_argument(
        "batch_size",
        type=int,
        help=HELPBATCH
    )
    train_parser.add_argument(
        "dataset_folder",
        type=str,
        help=HELPDATASET
    )
    
    # Argument OPTIONNEL (commence par des tirets)
    train_parser.add_argument(
        '-bc', '--batch_combined',
        dest="batch_combined", # Nom utilisé dans args.batch_combined
        type=int,
        default=None,
        help=HELPBATCHCOMBI
    )
    train_parser.add_argument(
        '-c',
        '--cross',
        action='store_true',
        default=False,
        help=HELPCROSS)

####################################################################################
####################################################################################
    create_embedded_parser = subparsers.add_parser("gen_gallery", help=HELPINITEMBEDDING)

    # Arguments positionnels (obligatoires)
    create_embedded_parser.add_argument(
        "model_path",
        type=str,
        help=HELPMODEL
    )

    create_embedded_parser.add_argument(
        "dataset_folder",
        type=str,
        help=HELPDATASET
    )
    create_embedded_parser.add_argument(
        '-c',
        '--cross',
        action='store_true',
        default=False,
        help=HELPCROSS)

####################################################################################
####################################################################################
    use_model = subparsers.add_parser("get_closest", help=HELPUSEMODEL)

    # Arguments positionnels (obligatoires)

    use_model.add_argument(
        "model_path",
        type=str,
        help=HELPMODEL
    )

    use_model.add_argument(
        "embed_path",
        type=str,
        help=HELPEMBEDING
    )

    use_model.add_argument(
        "image_path",
        type=str,
        help=HELPIMAGE
    )
    use_model.add_argument(
        "coordinates_path",
        type=str,
        help=HELPCOORDINATES
    )

    use_model.add_argument(
        '-c',
        '--cross',
        action='store_true',
        default=False,
        dest="cross",
        help=HELPCROSS
    )
####################################################################################
####################################################################################
    pca = subparsers.add_parser("pca", help=HELPPCA)

    pca.add_argument(
        "embed_path",
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
        "embed_path",
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
        "embed_path",
        type=str,
        help=HELPEMBEDING
    )

####################################################################################

    args = parser.parse_args()

    print(str(args))

    if args.subcommand == "train":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        images_path = Path(args.dataset_folder).joinpath("img")
        sat_path = Path(args.dataset_folder).joinpath("sat")

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
        if args.cross:
                if not sat_path.exists():
                    print(f"Erreur : Le dossier {sat_path} n'existe pas.")
                    return
                from src.cross import train_cross
                train_cross.train_cross(
                nb_epoch=args.nb_epoch,
                batch_size=args.batch_size,
                batch_combined=batch_combined,
                data_json=str(json_path.resolve()),
                data_images=str(images_path.resolve()),
                data_sat=str(sat_path.resolve())
            )
        else:
            from src import train_base
            train_base.train_base(
                nb_epoch=args.nb_epoch,
                batch_size=args.batch_size,
                batch_combined=batch_combined,
                data_json=str(json_path.resolve()),
                data_images=str(images_path.resolve())
            )

    elif args.subcommand == "gen_gallery":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        images_path = Path(args.dataset_folder).joinpath("img")
        model_path = Path(args.model_path)
        if not images_path.exists():
            print(f"Erreur : Le dossier {images_path} n'existe pas.")
            return

        if not json_path.exists():
            print(f"Erreur : Le fichier {json_path} n'existe pas.")
            return

        if not model_path.exists():
            print(f"Erreur : Le fichier {model_path} n'existe pas.")
            return
        from src import embed_database

        if args.cross:
            from src.cross import model_cross
            r_model = model_cross.CrossEncoder().to(embed_database.DEVICE)
        else:
            from src import model
            r_model = model.MixedEncoder().to(embed_database.DEVICE)
        r_model.load_state_dict(torch.load(str(model_path.resolve())))
        print(f"Dataset : {Path(args.dataset_folder)}")
        print(f"Model : {model_path}")

        embed_database.create_embeddings(
            r_model,
            str(images_path.resolve()),
            str(json_path.resolve())
        )

        print("Embeddings calculés et sauvegardés dans embeddings_db.pt")


    elif args.subcommand == "get_closest":
        embed = Path(args.embed_path)
        model_path = Path(args.model_path)
        image = Path(args.image_path)
        coordinates = Path(args.coordinates_path)
    
        if not image.exists():
            print(f"Erreur : Le fichier {image} n'existe pas.")
            return

        if not embed.exists():
            print(f"Erreur : Le fichier {embed} n'existe pas.")
            return

        if not model_path.exists():
            print(f"Erreur : Le fichier {model_path} n'existe pas.")
            return

        print(f"Embeddings : {embed}")
        print(f"Model : {model_path}")
        print(f"Image : {image}")
        from src import embed_database

        if args.cross:
            from src.cross import model_cross
            r_model = model_cross.CrossEncoder().to(embed_database.DEVICE)
        else:
            from src import model
            r_model = model.MixedEncoder().to(embed_database.DEVICE)
        r_model.load_state_dict(torch.load(str(model_path.resolve())))

        print(f"Model : {model_path}")
        print(f"Embeddings : {embed}")
        print(f"Image : {image}")

        embed_database.get_closest_locations(
            r_model,
            str(embed.resolve()),
            str(image.resolve()),
            str(coordinates.resolve())
        )

    elif args.subcommand in "tsne":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        embed = Path(args.embed_path)
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
        embed = Path(args.embed_path)
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
        embed = Path(args.embed_path)
        if not embed.exists():
            print(f"Erreur : Le fichier {embed} n'existe pas.")
            return

        print(f"Embeddings : {embed}")

        embed_database.pca(
            embed
        )


if __name__ == "__main__":
    main()