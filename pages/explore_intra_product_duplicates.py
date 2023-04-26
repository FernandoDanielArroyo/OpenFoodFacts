import streamlit as st
import json
from PIL import Image

def read_automatic_analysis():
    with open('data/automated_duplicated.json') as f:
        data = json.load(f)
    return data

@st.cache_data
def load_products_name():
    with open('data/label_of_products.json') as f:
        product_data = json.load(f)
    return product_data

def get_selected_category(index, categories):
    selected_category = categories[st.session_state['selected_category']]
    return selected_category
if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = 0

data = read_automatic_analysis()
product_data = load_products_name()
categories = list(data.keys())
selected_category = get_selected_category(st.session_state['selected_category'], categories)
product_name = product_data[selected_category]['product_name']
st.write(f'{selected_category=}')
st.write(f'{product_name=}')

cols =  st.columns([2,2,1])
next = cols[0].button(label='Next')
previous = cols[1].button(label='Previous')

if next:
    st.session_state['selected_category'] += 1

if previous:
    if st.session_state['selected_category'] > 0:
        st.session_state['selected_category'] += 1

for img_1_name, img_2_name, distance in data[selected_category]:
    labels = [img_1_name, img_2_name]
    
    img_1 = Image.open(f'data/img/{img_1_name}.jpg')
    img_2 = Image.open(f'data/img/{img_2_name}.jpg')
    images = [img_1, img_2]
    st.write(distance)
    st.image(images, labels)