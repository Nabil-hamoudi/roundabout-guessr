# Position Guesseur

*Alexandre DUCROS, Nabil HAMOUDI*

## Presentation

- Machine Learning project starting from december 2025 to January 2026.
- The objective of this project is to retrieve the coordinates of where a photo was taken.

### Requirement

> Python 3.14.2 tqdm

#### Model Training

> torch torchvision numpy matplotlib opencv-python albumentations random-fourier-features-pytorch geopy

#### DataSet Scraping

> streetlevel

## Quick install

```sh
py -m pip install tqdm torch torchvision numpy matplotlib opencv-python albumentations random-fourier-features-pytorch geopy
```

## Operation

```
Position Guesseur
│   README.md
│   main.py  
│
└───src\
│   │   dataset.py
│   │   embed_database.py
│   │   model.py
│   │   train_clip.py
│   │ 
│   └───cross\
│       │   dataset_cross.py
│       │   model_cross.py
│       │   train_cross.py
│  
└───dataset\  
    │   coordinates.json
    │
    └───data_sat\ [OPTIONAL]
    │    │ sat_0.jpg
    │    │ dat_1.jpg
    │    │ ...
    │  
    └───data\
        │ img_0.jpg
        │ img_1.jpg
        │ ...
```

- *main.py* : CLI to use our model, handle training, and generate all embeddings.
- *model.py* : Training of the model; generates *model_x.pt* at every epoch.
- *embed_database.py* : Generation and visualization of embeddings.
  - *We can visualize via t-SNE, PCA, and PCA combined with geographical data*
- *dataset\\* : Contains the dataset with *coordinates.json*, *data\\* (images) and *data_sat\\* (satellite images) this images are use for cross model there are optional. This folder does not exist in the repo but is expected here for certains application.
- *cross\\* : Contains the code for cross model.

## Dataset

> Our dataset is scraped from Google Street View. We pick random coordinates in certain places and use the library *streetlevel* to get the Google Panorama ID.
> With each panorama ID, we retrieve the panorama image and extract images from any angle we want.

- All images have a corresponding entry in "coordinates.json" containing data/annotations.

Using this method, we generated multiple datasets for testing and results. The scraped locations are:

- France
  - Dataset: 70,000 images randomized, concentrated in the 50 biggest cities of France.
    - https://drive.google.com/file/d/1YalWVF-CK_d6iy4c440gwfKzVtNhZHmN/view?usp=drive_link
  - Dataset: 300,000 images randomized across all of France.
    - https://drive.google.com/file/d/1m-bu9LzyZ_dO6VezFXGeoO7Kp5gZKqbe/view?usp=drive_link
- Paris
  - xxxxxxxxxx
    - xxxxxxxxxx

> Every dataset folder for our project must have a *coordinates.json* file and a *data\\* folder containing images named with the prefix *img_* followed by their *id* and *extension* (optionally the prefix *sat_* followed by their *id* and *extension* ). For each images we can have Satelite images in *data_sat\\* if we use the cross model.

- For example *img_0.jpg* (optionally *sat_0.jpg* the satelite image linked to it).
  In *coordinates.json* for each image the data is formed this way:

```json
 "img_id :"{
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


| Field            | Type    | Description                                          |
| ---------------- | ------- | ---------------------------------------------------- |
| `longitude`      | float   | Original longitude coordinate in EPSG:4326 system    |
| `latitude`       | float   | Original latitude coordinate in EPSG:4326 system     |
| `pano_id`        | string  | Google Street View panorama ID                       |
| `sampled_lon`    | float   | Sampled longitude for processing in EPSG:4326 system |
| `sampled_lat`    | float   | Sampled latitude for processing in EPSG:4326 system  |
| `view_direction` | string  | Viewing direction (front/left/right/back)            |
| `h_deg`          | integer | Horizontal angle in degrees [0 - 360]                |
| `v_deg`          | integer | Vertical angle in degrees [0 - 360]                  |

## CLI

```sh
python main.py <commande> [arguments]
```

Every commands use an application of our project:

- *train* : Train the model with a dataset folder.
  - ```main.py train [-bc BATCH_COMBINED] nbr_epoch batch_size dataset_folder```
    *BATCH_COMBINED* is used for instances with insufficient VRAM for larger batch sizes, as the batch size impacts loss calculations.
- *embedgen* : Generate embeddings using a *model_file* and *dataset_folder*.
- *model* : Use a *model_file* and *embed_file* to predict the coordinates of an image.
- *pca* : Visualize the PCA from the embeddings
- *pca_geo* : Visualize the PCA from the embeddings with geographiques coordinnates
- *tsne* :  Visualize the TSNE from the embeddings

> With the argument *cross* we can change the model we use between the base model and cross model.

## Models

We have generated differents models:

- France
  - Models with 70 000 images randomised concentrated in the 50 biggest city of France.
    - xxxxxxxxxxxxxxxx
- Paris
  - xxxxxxxxxx
    - xxxxxxxxxx

The generated models vary; the model attached to Europe was part of a simpler method than France, and the ones from France are simpler again and dont use satelite images for the training.

## Sources

- Vicente Vivanco Cepeda, Gaurav Kumar Nayak and Mubarak Shah (2023). GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization. https://arxiv.org/abs/2309.16020
