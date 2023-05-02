# Detecting duplicates using images features

This project aims to resolve the following github issue : https://github.com/openfoodfacts/openfoodfacts-ai/issues/203.

We want to detect duplicated products from images.  
We identified the following tasks:
- identify similar images in each product
- identity similar images in different product (duplicated products) 

## Env

use conda and env.yaml 

## Data

We started by working on a subset of the full database (~10k products and 33k images):
Id of product codes are available in the images_downloaded.txt in data/.

## Features

Features from images were extracted using huggingface implementation of visual transformers (ViT, Swin).
Embeddings are able to be extracted using the embedings.py script
```
python embedings.py google/vit-base-patch16-224
python embedings.py microsoft/swin-tiny-patch4-window7-224
```
Limitations: 
- only works images per images (no batching)
- exceptions (examples images not in 3 channels) are excluded
- only tested with the following models:
    - google/vit-base-patch16-224
    - microsoft/swin-tiny-patch4-window7-224

## Detection of duplicates products

Similarity was proposed using cosine distance.

When exploring similarity of images, we observed that duplicates of products were able to be detected but images containing nutritional information/bar code  were not able to be separated correctly. We used clustering to try to separate between images of products and images of nutritional labels/barcodes.


### Clustering : Removing labels 
To separate the two categories (products and nutritional information), we used a simple clustering (kmeans, 2 clusters) based on vision embeddings.
Embedding on a subsets of images are able to be observed using tensorboard (observed on a subset of 1k images)

```
python build_tensorboard_visu.py
tensorboard --logdir embedding_visu_log
```
If tensorboard does not display data please select 'projector' on the drop-down menu at the top right.   
We observe that class 0 is mainly nutritional labels/bar codes and class 1 mainly products.


## Application to facilitate removal of duplicates products

We decided to use a streamlit web app to facilitate user detection of duplicated products.

```
streamlit run app.py
```

List of applications available:
- app : explore close images in different products (no order)
- intra product duplicates : explore closes images in the same product (no order)

- explore inter product duplicates : images pair are ordered by the cosine distance (**run automatic_inter_detection before**)
- explore intra product duplicates : for each product show detected images pair (**run automatic_intra_detection before**)

Limitations:
- code for this part is not commentated
- user inputs are not saved and not consistent between applications
- naming of different pages is bad
- visual bug when changing images using buttons 
