import streamlit as st
from PIL import Image
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

st.set_page_config(layout="wide")

@st.cache_data
def list_images(embedding_df):
    list_img = embedding_df.index.to_list()
    return list_img

@st.cache_data
def load_products_name():
    with open('data/label_of_products.json') as f:
        product_data = json.load(f)
    return product_data

@st.cache_data
def load_embedding_data(selected_embedding_file):
    df = pd.read_csv(selected_embedding_file, index_col=0).astype('float16')
    df = df[df['label'] == 1]
    return df

@st.cache_resource
def get_cosine_distances(embedding_data):
    if 'label' in embedding_data.columns:
        df = embedding_data.drop(columns='label')
    else:
        df = embedding_data.copy()
    cosines = cosine_distances(df)
    cosines = pd.DataFrame(cosines, index=df.index, columns=df.index)
    return cosines


product_data = load_products_name()
embedding_data = load_embedding_data('data/labels_vit_kmeans_2.csv')
embedding_data.index.name = 'image_name'
images = list_images(embedding_data)
cosines = get_cosine_distances(embedding_data)
if 'counter' not in st.session_state: 
    st.session_state['counter'] = 0

datas = [('image_list', images),
         ('product_data', product_data),
         ('embedding_data', embedding_data),
         ('cosines', cosines)]
for key, val in datas:
    if key not in st.session_state:
        st.session_state[key] = val

st.header('Deduplicating OpenfoodFact images')
st.subheader('Visualisation of close images')

choice_image = images[st.session_state['counter']]


#image columns
col1, col2 = st.columns(2)

if choice_image:
    category = choice_image.split('_')[0]
    product_name = product_data[category]['product_name']
    st.text('Index selected image: ')
    st.session_state['counter']
    image = Image.open(f'data/img/{choice_image}.jpg')
    with col1:
        st.text(product_name)
        st.image(image)

    query = cosines.loc[choice_image].sort_values()
    query.name = 'distance'
    query = query.reset_index()
    query['categories'] = query['image_name'].apply(lambda x: x.split('_')[0])
    query = query[query['categories']!= category].head(1)
    
    image = Image.open(f'data/img/{query.iloc[0, 0]}.jpg')

    category2 = query.iloc[0, 0].split('_')[0]
    product_name2 = product_data[category2]['product_name']

    with col2:
        st.text(product_name2)
        st.image(image)

st.dataframe(query)

# buttons
cols =  st.columns([3,3,1])
similar = cols[0].button(label='Similar')
not_similar = cols[1].button(label='Not similar')

if similar:
    st.session_state['counter'] += 1
    
if not_similar:
    st.session_state['counter'] += 1

if st.session_state['counter'] >= 1:
    previous = cols[2].button(label='previous')
    if previous:
        st.session_state['counter'] -= 1

choice_image = images[st.session_state['counter']] 


    