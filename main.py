import argparse
from pathlib import Path

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
HELPFR = "Utiliser les coordonnées de la France pour normaliser"
HELPSPIDER = "Génére la carte des erreurs pour un modele entrainer et un dataset donné"
HELPCARTEOUTPUT = "Chemin vers le fichier de sortie de la carte en html"
HELPNUMBERPOINTS = "Nombre de points à afficher sur la carte des erreurs"
HELPBENCHMARK = "Lance le benchmark sur un modèle, une base d'embeddings donnée, ses coordonnées et un test set (Paris datasets only !)"

def check_paths_exist(*paths: Path) -> bool:
    for p in paths:
        if not p.exists():
            print(f"Erreur : Le fichier ou dossier '{p}' n'existe pas.")
            return False
    return True

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
    train_parser.add_argument(
        '-fr',
        '--france',
        action='store_true',
        default=False,
        help=HELPFR)

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
    create_embedded_parser.add_argument(
        '-fr',
        '--france',
        action='store_true',
        default=False,
        help=HELPFR)

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
    use_model.add_argument(
        '-fr',
        '--france',
        action='store_true',
        default=False,
        help=HELPFR)
####################################################################################
####################################################################################
####################################################################################
####################################################################################
    benchmark_parser = subparsers.add_parser("benchmark", help=HELPBENCHMARK)
    # Arguments positionnels (obligatoires)
    benchmark_parser.add_argument(
        "model_path",
        type=str,
        help=HELPMODEL
    )
    benchmark_parser.add_argument(
        "embeds_coords_path",
        type=str,
        help=HELPEMBEDING
    )
    benchmark_parser.add_argument(
        "coordinates_path",
        type=str,
        help=HELPCOORDINATES
    )
    benchmark_parser.add_argument(
        "testset_folder",
        type=str,
        help=HELPDATASET
    )

    benchmark_parser.add_argument(
        '-c',
        '--cross',
        action='store_true',
        default=False,
        dest="cross",
        help=HELPCROSS
    )

####################################################################################

    args = parser.parse_args()

    print(str(args))

    if args.subcommand == "train":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        images_path = Path(args.dataset_folder).joinpath("img")
        sat_path = Path(args.dataset_folder).joinpath("sat")

        if not check_paths_exist(json_path, images_path):
            return

        print(f"Dataset : {Path(args.dataset_folder)}")
        print(f"Batch combined : {args.batch_combined}")

        if args.batch_combined is None:
            batch_combined = args.batch_size
        else:
            batch_combined = args.batch_combined

        # Lancer l'entraînement
        if args.cross:
                if not check_paths_exist(sat_path):
                    return
                from src.cross_view import train_cross
                train_cross.train_cross(
                nb_epoch=args.nb_epoch+1,
                batch_size=args.batch_size,
                batch_combined=batch_combined,
                data_json=str(json_path.resolve()),
                data_images=str(images_path.resolve()),
                data_sat=str(sat_path.resolve()),
                want_france=(args.france == True)
            )
        else:
            from src.base import train_base
            train_base.train_base(
                nb_epoch=args.nb_epoch+1,
                batch_size=args.batch_size,
                batch_combined=batch_combined,
                data_json=str(json_path.resolve()),
                data_images=str(images_path.resolve()),
                want_france=(args.france == True)
            )

    elif args.subcommand == "gen_gallery":
        json_path = Path(args.dataset_folder).joinpath("coordinates.json")
        images_path = Path(args.dataset_folder).joinpath("img")
        model_path = Path(args.model_path)

        if not check_paths_exist(json_path, images_path, model_path):
            return

        from src.embed_database import init_model, create_embeddings

        create_embeddings(
            init_model(model_path, args.cross, args.france),
            str(images_path.resolve()),
            str(json_path.resolve())
        )

        print("Embeddings calculés et sauvegardés dans embeddings_db.pt")


    elif args.subcommand == "get_closest":
        embed = Path(args.embed_path)
        model_path = Path(args.model_path)
        image = Path(args.image_path)
        coordinates = Path(args.coordinates_path)

        if not check_paths_exist(embed, model_path, image, coordinates):
            return
        from src.embed_database import init_model, get_closest_locations


        print(f"Model : {model_path}")
        print(f"Embeddings : {embed}")
        print(f"Image : {image}")

        get_closest_locations(
            init_model(str(model_path.resolve()), args.cross, args.france),
            str(embed.resolve()),
            str(image.resolve()),
            str(coordinates.resolve())
        )


if __name__ == "__main__":
    main()