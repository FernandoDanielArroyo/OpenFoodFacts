import streamlit as st
from PIL import Image

if 'counter_categories' not in st.session_state: 
    st.session_state['counter_categories'] = 0

images = st.session_state['image_list']
product_data = st.session_state['product_data']
cosines = st.session_state['cosines']
images_embedding = st.session_state['embedding_data'].reset_index()
images_embedding['categories'] = images_embedding['image_name'].apply(lambda x: x.split('_')[0])
images_embedding = images_embedding.set_index('categories')

categories = sorted(set(images_embedding.index))
# choice_image = st.selectbox(label='Select an image', options=images, index=0)

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

all_images_of_category = images_embedding.loc[choice_category][['image_name']]

images = []
if len(all_images_of_category) > 1:
    for row in all_images_of_category.sort_values('image_name').itertuples():
        images.append(Image.open(f'data/img/{row.image_name}.jpg'))
else:
    images.append(Image.open(f'data/img/{all_images_of_category["image_name"]}.jpg'))
st.image(images)