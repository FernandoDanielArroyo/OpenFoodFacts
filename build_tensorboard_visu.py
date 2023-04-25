# %%
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image
from transformers import AutoFeatureExtractor

# %%
path_log = Path('embedding_visu_log')
image_folder = Path('data/img/')
path_embedding_with_labels = Path('data/labels_vit_kmeans_2.csv')
path_log.mkdir(exist_ok=True)
model_ckpt = 'google/vit-base-patch16-224'

# %% [markdown]
# # Load embeddings

# %%
df = pd.read_csv(path_embedding_with_labels, index_col=0)
df.index.name = 'image_id'
df = df.sample(n=1000, random_state=42)

# %%
df

# %% [markdown]
# # Load data : images, embedding, labels

# %%
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
images = []
for image_name in df.index:
    path_image = image_folder / f'{image_name}.jpg'
    image = Image.open(path_image)
    image = extractor(image, return_tensors="pt")['pixel_values'][0]
    images.append(image)
images = torch.stack(images, axis=0)

# %%
labels = df['label']
embeddings = torch.from_numpy(df.drop(columns='label').values)

# %% [markdown]
# # Write data to tensorboard

# %%
writer = SummaryWriter(path_log)

# %%
writer.add_embedding(mat=embeddings, metadata=labels, label_img=images)


