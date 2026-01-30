# 🥐🗼 Position Guessr  ![Python](https://img.shields.io/badge/python-3.14-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)

**Authors**: Alexandre DUCROS, Nabil HAMOUDI

## 📝 Overview

**Position Guessr** is a Deep Learning project aiming to retrieve the precise geographical coordinates (latitude, longitude) of a location based solely on a street-level photograph.

Inspired by the [GeoCLIP](https://arxiv.org/abs/2309.16020) architecture, our model aligns visual features (from streetview images and satellites images) with spatial features (coordinates) using Contrastive Learning.

## ⚙️ Installation

### Requirements

- Python 3.14+
- CUDA-capable GPU (Recommended for training)
- Pytorch installed

#### Essential dependencies (other than Pytorch)

> torchvision numpy matplotlib opencv-python albumentations random-fourier-features-pytorch geopy gdown

#### Scraping dependencies

> streetlevel

### Quick Start

Clone the repository and install the dependencies :

```sh
py -m pip install tqdm torchvision numpy matplotlib opencv-python albumentations random-fourier-features-pytorch geopy gdown folium
```

### For an even quicker start :

Use our [demo](https://colab.research.google.com/drive/1z9V7xi33NAnsBgNWSKV_DYNNnzjHAKVb?usp=sharing) ! It has a dataset downloaded, a trained model, generated embeddings. Life is good !

## 🚀 Usage

The project is designed to be run through a CLI, here it is main.py.

Across the CLI the `--cross` (or `-c`) option is to use the cross-view model, note that you then need a dataset with satellies views. The `--france` (or `-c`) option is to tell the model that you are in a national context and not focused on Paris.

#### 1. Training

To train the model on a specific dataset. Use `batch_combined` (or `bc`) to simulate large batches on limited VRAM. Note that it still use heavy amount on VRAM and that 12 to 16Gb are recommanded to train the model.

```
python main.py train [-bc BATCH_ACCUMULATION] <epochs> <batch_size> <dataset_path>

# Example (Training the Base Model)
python main.py train -bc 256 150 32 ./dataset/paris

# Example (Training the Cross View Model)
python main.py train -bc 256 150 32 ./dataset/paris --cross
```

#### 2. Inference & Visualisation

```bash
python main.py get_closest [-c] [-fr] model_path embed_path image_path coordinates_path #Inference
python main.py gen_gallery [-c] [-fr] model_path dataset_folder #Embed database generation
python main.py placehold for Visualisation
python main.py --help #To get infos about other commands, can also be applied to commands

#Inference example using cross-view
python main.py get_closest -c model_paris_50k.pt embeddings_paris_50k.pt ./test_img.jpg ./dataset/coordinates.jpg 
```

## 📂Project Structure

```
Position Guesseur
├── main.py                     # CLI Entry point
├── src/
│   ├── base/                   # Standard Model (Image <-> Location)
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train_base.py
│   │
│   ├── cross_view/             # Cross-View Model (+ Satellite)
│   │   ├── dataset_cross.py
│   │   ├── model_cross.py
│   │   └── train_cross.py
│   │
│   ├── model_components/       # Shared Architectures
│   │   ├── image_encoder.py    # DINOv2 backbone + Adapters
│   │   └── location_encoder.py # RFF + ResMLP
│   │
│   ├── embed_database.py       # Embedding generation & Visu
│   ├── training_utils.py       # Loss function & Dataset splitting
│   ├── validation_utils.py     # Validation execution
|   └── visu_spider.py          # Spider thread like visualisation of predicted position
|   
│
└── dataset/                    # Data storage
    ├── coordinates.json        # Labels and Metadata
    ├── img/                   # Street View Images
    └── sat/               # Satellite Images (Optional, obligatory for cross-view)
```

## 🧠 Methodology

Our approach relies on **Deep Metric Learning**:

1. **Image Encoder:** We use **DINOv2 (Small)** frozen weights with trainable adapters to extract semantic features from streetview images and satellite images.
2. **Location Encoder:** We use **Random Fourier Features (RFF)** to map 2D GPS coordinates into a high-dimensional space. This structures the encoder as a **Neural Feature Field**, enabling the model to learn high-frequency spatial details rather than smooth global trends.
3. **Loss Function:** We utilize a **Masked InfoNCE Loss**.
   * It maximizes similarity between an image and its location.
   * It treats physically close locations (e.g., < 10m) as valid positives (masking) to avoid false negatives during contrastive learning.

You can learn more about the architecture in the "model_explained.ipynb" or only [here](https://colab.research.google.com/drive/1VPylq210Usa3KIG8PUrhk0AHZs1yfn7l?usp=sharing) !

## 💾 Datasets

> Our data is scraped from Google Street View. Each dataset must contain a `coordinates.json` file and an image folder. We pick random coordinates in certain places and use the library *streetlevel* to get the Google Panorama ID and image.

### Available Datasets

* **France (Urban Focus):** 70k images from french 50 cities. [Download](https://drive.google.com/file/d/1VElOIWDLL83oL-OrIfO-i7G07vkbTfpn/view?usp=sharing)
* **France (Global):** 300k images randomized across the country. [Download](https://drive.google.com/file/d/1PQ7r9Ijj5XKESN2vsECv_YgFcfE7xzAh/view?usp=sharing)
* **Paris 50K :** 50K images randomized across the Parisian Region (includes Satellite views). [Download](https://drive.google.com/file/d/1Ht602iXoHgHuJ9hNJh9biDCdQwxGDzPH/view?usp=drive_link)

Note that Paris 50K dataset, model and embeddings are trained using satellite images and then need the `--cross` (or `-c`) option in the CLI.

> Every dataset folder indeed needs to have a coordinates.json (detailled below), a folder named "img" where the streetview images are and optionally a "sat" folder where the corresponding satellite images goes.
>
> The id should be shared across the folders (even if the prefix_ is not the same).

```json
 "img_id :"{
    "longitude": float, // Longitude in EPSG:4326 system
    "latitude": float, // Latitude in EPSG:4326 system
    "pano_id": string, // ID of the streetview panorama
    // other keys are not necessary for current models to run.
}
```

## 🤖 Models

Paris 50K model we stopped early : [Download](https://drive.google.com/file/d/1tcvvOx-qeKgNDOw9BItXGN3eOlYzIB1_/view?usp=sharing)

## 📊 Results (Performance)

Here are the retrieval results on our validation set (Paris 50K dataset) after 13 epochs:

- R@100m : 20%
- R@250m : 23.77%
- R@1km : 36.27%
- R@2km : 45.80%

With a median of 2.5km, outlying the ability of the model to anchor itself on distinct images (monuments), but having a hard time to differentiate similar streets. Note that this is only a 13 epochs training.

## 📚 References

- **GeoCLIP:** Vicente Vivanco Cepeda, Gaurav Kumar Nayak and Mubarak Shah (2023). *GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization.*[arXiv:2309.16020](https://arxiv.org/abs/2309.16020)
- **DINOv2:** Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
