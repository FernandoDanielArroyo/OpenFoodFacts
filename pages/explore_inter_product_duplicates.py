import streamlit as st
import json
from PIL import Image

def read_automatic_analysis():
    with open('data/inter_product_duplicated.json') as f:
        data = json.load(f)
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1][1]))
    return sorted_data

@st.cache_data
def load_products_name():
    with open('data/label_of_products.json') as f:
        product_names = json.load(f)
    return product_names

data = read_automatic_analysis()
image_list = list(data.keys())
product_names = load_products_name()

if 'selected_product' not in st.session_state:
    st.session_state['selected_product'] = 0

selected_image = image_list[st.session_state['selected_product']] 
category = selected_image.split('_')[0]
product_name = product_names[category]['product_name']
st.write(f'{selected_image=}')
st.write(f'{product_name=}')

closest_img_name, distance = data[selected_image]
closest_img_category = closest_img_name.split('_')[0]
closest_product_name = product_names[closest_img_category]['product_name']
img_1 = Image.open(f'data/img/{selected_image}.jpg')
img_2 = Image.open(f'data/img/{closest_img_name}.jpg')
st.write(f'{closest_product_name=}   {distance=:.3f}')
st.image([img_1, img_2])

cols =  st.columns([2,2,1])
next = cols[0].button(label='Next')
previous = cols[1].button(label='Previous')

if next:
    st.session_state['selected_product'] += 1
if previous:
    if st.session_state['selected_product'] >0:
        st.session_state['selected_product'] -= 1