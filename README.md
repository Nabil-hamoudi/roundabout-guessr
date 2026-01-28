# Position Guesseur

*Alexandre DUCROS, Nabil HAMOUDI*

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
│   
└───dataset\
    │   coordinates.json
    │
    └───data\
        │ image
        │ img_00000.jpg
        │ img_00001.jpg
        │ ...
```

 - *main.py* : CLI to use our model, training and generations of all the embeddings.
 - *model.py* : Training of the model generate model_x.pt the model at every epoch 
 - *embed_database.py* : Generation of the embedding and visualisation of them
   - *We can visualise the TSNE, PCA and PCA with the geographical data of the dataset*
 - *dataset\\* : Contain the dataset with the *coordinates.json* and *data\\* all our images. 
## Dataset
 > Here our dataset is scrapped from GoogleStreetView, we take random coordinate in France and use the librairy Streetlevel to get the google panorama ID.
 > With each panorama ID we can get the panorama image and with each panorama we take images from any angle we want.
 - For all the images we have "coordinates.json" containing all the data/annotation associated with each images.


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


## Sources

- Philipp Lindenberger and Paul-Edouard Sarlin and Jan Hosang and Matteo Balice and Marc Pollefeys and Simon Lynen and Eduard Trulls (2025). Scaling Image Geo-Localization to Continent Level. https://arxiv.org/abs/2510.26795

- Vivanco Cepeda, V., Nayak, G. K., & Shah, M. (2023). GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization. https://arxiv.org/abs/2309.16020
