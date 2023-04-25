import streamlit as st
from PIL import Image
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

st.set_page_config(layout="wide")

@st.cache_data
def list_images():
    path_images = Path('data/img')
    list_img = list(path_images.rglob('*.jpg'))
    return list_img

@st.cache_data
def load_products_name():
    with open('data/label_of_products.json') as f:
        product_data = json.load(f)
    return product_data

@st.cache_data
def load_embedding_data(selected_embedding_file):
    df = pd.read_csv(selected_embedding_file, index_col=0)
    return df

@st.cache_resource
def get_cosine_distances(embedding_data):
    if 'label' in embedding_data:
        df = embedding_data.drop('label')
    else:
        df = embedding_data.copy()
    cosines = cosine_distances(df)
    cosines = pd.DataFrame(cosines, index=df.index, columns=df.index)
    return cosines

images = list_images()
product_data = load_products_name()
embedding_data = load_embedding_data('data/vit-base-patch16-224.csv')
embedding_data.index.name = 'image_name'
cosines = get_cosine_distances(embedding_data)
st.header('Exploration of OpenfoodFact using Product Image embeddings')

choice_image = st.selectbox(label='Select an image', options=images)
if choice_image:
    product_name = product_data[choice_image.stem.split('_')[0]]['product_name']
    st.write('Selected image')
    st.write(f'{product_name=}')
    image = Image.open(choice_image)
    st.image(image)

    st.header('Visualisation of close images')
    # st.dataframe(embedding_data.head())
    # st.dataframe(embedding_data.loc[choice_image.stem])
    top_5 = cosines.loc[choice_image.stem].sort_values().head(6)
    images = []
    captions = []
    for index, distance in top_5.items():
        images.append(Image.open(f'data/img/{index}.jpg'))
        product_name = product_data[index.split('_')[0]]['product_name']
        captions.append(f'{product_name}_{distance}')
    st.image(images, captions)
    # st.dataframe(cosines.loc[choice_image.stem].sort_values().head(5))