import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import numpy as np

if 'counter_categories' not in st.session_state: 
    st.session_state['counter_categories'] = 0

@st.cache_resource
def get_cosine_distances(embedding_data):
    if 'label' in embedding_data:
        df = embedding_data.drop('label')
    else:
        df = embedding_data.copy()
    cosines = cosine_distances(df)
    cosines = pd.DataFrame(cosines, index=df.index, columns=df.index)
    return cosines


images = st.session_state['image_list']
product_data = st.session_state['product_data']
cosines = st.session_state['cosines']
images_embedding = st.session_state['embedding_data'].reset_index()
images_embedding['categories'] = images_embedding['image_name'].apply(lambda x: x.split('_')[0])
images_embedding = images_embedding.set_index('categories')

categories = sorted(set(images_embedding.index))
choice_category = categories[st.session_state['counter_categories']]
next = st.button('Next category')
previous = None
if st.session_state['counter_categories'] > 0:
    previous = st.button('Previous Category')
if next:
    st.session_state['counter_categories'] += 1
if previous:
    st.session_state['counter_categories'] -= 1


st.write(f'{st.session_state["counter_categories"]}')
product_name = product_data[choice_category]['product_name']
st.write(f'Product name : {product_name}  Category id : {choice_category}')

# Get all images from the selected category
all_images_of_category = images_embedding.loc[choice_category]
# st.dataframe(all_images_of_category)
show_all = st.radio('show_all', options=[True, False], horizontal=True)
if show_all:
    images = []
    if type(all_images_of_category) == pd.DataFrame:
        for row in all_images_of_category.sort_values('image_name').itertuples():
            images.append(Image.open(f'data/img/{row.image_name}.jpg'))
    else:
        images.append(Image.open(f'data/img/{all_images_of_category.loc["image_name"]}.jpg'))
    st.image(images)

st.header('Duplicates within the same categories')
if type(all_images_of_category) == pd.DataFrame:
    all_images_of_category = all_images_of_category.set_index('image_name')
    cosine_distances = get_cosine_distances(all_images_of_category)
    # st.dataframe(cosine_distances)
    cosine_distances.index.name = None
    cosine_distances = cosine_distances.where(np.triu(np.ones(cosine_distances.shape)).astype(bool))
    # st.dataframe(cosine_distances)
    cosine_distances = cosine_distances.stack()
    cosine_distances = cosine_distances[(cosine_distances > 0) & (cosine_distances < 0.2)]
    # st.dataframe(cosine_distances)
    # index_list = cosine_distances.index
    # st.write(index_list)
    for index, distance in cosine_distances.items():
        caption = [index[0], index[1]]
        img_1 = Image.open(f'data/img/{index[0]}.jpg')
        img_2 = Image.open(f'data/img/{index[1]}.jpg')
        st.write(distance)
        st.image([img_1, img_2], caption=caption)
