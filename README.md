# roundabout-guessr

Projet machine learning 2025-2026

Finir de DL les données sur Google Street View

Mettre les données dans un dossier data : important !!

Ajouter le train en fonction de la distance des coordonnées ?

-> Un poids sur la loss en fonction de la distance pq pas franchement


Le projet est organisé en deux parties : L'embedding et le retrieval

L'embedding est un simple encodeur, qui prend une image (normalisée selon Imagenet) en 600x400 RGB.
L'espace latent/d'embedding est un vecteur de taille 128.

Le retrieval se base sur une base de données, créée sur la prédiction des toutes les images du train set. On prédit alors le vecteur d'embed de l'image et on compare par rapport aux vecteurs de la base de données.


L'entraînement de l'embedding : Il se passe sur le metric deep learning.

On utilise comme loss une triple margin avec facteur de distance.
La triple margin demande 3 images, une ancre, une image positive et une négative. Le but étant de se demander si l'image positive est plus proche de l'ancre ou pas, à marge près. On y ajoute une composante "distance", ajustant la marge en fonction de la distance et permettant d'apprendre un espace "continu" et permettant au réseau de se faire sa propre carte mentale.

Pour le test de validation l'on prend 40 rond points au hasard et on y prend une image de chaque. On construit la BD pour chaque étape de validation et on regarde l'accuracy de prédiction.

Pour l'entraînement on prend le reste et on y applique une procédure supervisée standard.


Pour le retrieval :

On applique un algorithme naïf, regarder FAISS etc etc.
