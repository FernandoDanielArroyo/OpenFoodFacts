# Detecting duplicates using images features

This project aims to resolve the following github issue : https://github.com/openfoodfacts/openfoodfacts-ai/issues/203.

We want to detect duplicated products.

## Data

We started by working on a subset of the full database (~10k products and 33k images):
Index of used product codes are available in the images downloaded txt.

## Features

Features from images were extracted using :huggingface: implementation of visual transformers (ViT, Swin).
Embeddings are able to be extracted using the embedings.py script
```
python embedings.py $HUGGINGFACE_MODEL
```
Limitations: 
- only works images per images (no batching)
- exceptions (images not in 3 channels) are excluded
- only tested with the following models:
    - google/vit-base-patch16-224
    - microsoft/swin-tiny-patch4-window7-224

## Detection of duplicates products

Similarity was proposed using cosine distance.

When exploring similarity of images, we observed that duplicates of products were able to be detected but images containing nutritional information were not able to be separated correctly.

add examples img

If locahost does not display data please select 'projector' on the drop-down menu at the top right.   

### Clustering : Removing labels 
To separate the two categories (products and nutritional information), we used a simple clustering based on vision embeddings.
Embedding on a subsets of images are able to be observed using tensorboard and build_tensorboard_label.py
 
```
python build_tensorboard_visu.py
tensorboard --logdir embedding_visu_log
```

add examples img

## Application to facilitate removal of duplicates products

We decided to use a streamlit web app to facilitate user detection of duplicated products.

```
streamlit run app.py
```
For each images in the dataset, 
we present the user with the closest match (smallest cosine distance) of other images of a different categories.
The user is showed more information concerning the products :
- product names
- number of images by product
- food categories

The user is able to validate that the products are similar and the next pair of images are presented.
Data entered by the user should be stored (which format ?)
