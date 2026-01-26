# Position Guesseur

*Nabil HAMOUDI, Alexandre DUCROS*

## Presentation

 -  Machine Learning project starting from december 2025 to January 2026.
 - This project objective is to get from a photo the coordinate of where it has been taken.

### Requirement

> Python 3.14.2 tqdm

#### Model Training

> torch numpy matplotlib sklearn

#### DataSet Scrapping

> streetlevel 

## Operation
 - dataset\
    - gen.py
 - src\

## Dataset
 > Here our dataset is scrapped from GoogleStreetView, we take random coordinate in France and use the librairy Streetlevel to get the google panorama ID.
 > With each panorama ID we can get the panorama image and with each panorama we take images from any angle we want.

 - Parle des 2 maniere de recup? genre directement avec l'image de facon random ou avec les villes etc?
 - For all the images we have "coordinates.json" containing all the data connected/annotation associated with each images.


```json
 "image name :"{
    "longitude": float,
    "latitude": float,
    "pano_id": string,
    "sampled_lon": float,
    "sampled_lat": float,
    "view_direction": string,
    "h_deg": integer,
    "v_deg": integer
}
```

| Field | Type | Description |
|-------|------|-------------|
| `longitude` | float | Original longitude coordinate |
| `latitude` | float | Original latitude coordinate |
| `pano_id` | string | Google Street View panorama ID |
| `sampled_lon` | float | Sampled longitude for processing |
| `sampled_lat` | float | Sampled latitude for processing |
| `view_direction` | string | Viewing direction (front/left/right/back) |
| `h_deg` | integer | Horizontal angle in degrees |
| `v_deg` | integer | Vertical angle in degrees |


## Notre Réseaux de Neurones

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




TODO : Ajouter visualisation sur la répartition des coordonnées et sur la répartitions de l'espace latent (PCA/T-SNE)

## Sources

- Lindenberger, P., Sarlin, P.-E., Hosang, J., Balice, M., Pollefeys, M., Lynen, S., & Trulls, E. (2025). Scaling Image Geo-Localization to Continent Level. *arXiv preprint arXiv:2510.26795*. https://arxiv.org/abs/2510.26795

- Vivanco Cepeda, V., Nayak, G. K., & Shah, M. (2023). GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization. *arXiv preprint arXiv:2309.16020*. https://arxiv.org/abs/2309.16020
