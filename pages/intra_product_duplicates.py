import streamlit as st
from PIL import Image

images = st.session_state['image_list']
product_data = st.session_state['product_data']
cosines = st.session_state['cosines']
categories = [image.stem.split('_')[0] for image in images]
categories = set(categories)

choice_image = st.selectbox(label='Select an image', options=images, index=0)
if choice_image:
    categorie = choice_image.stem.split('_')[0]
    product_name = product_data[categorie]['product_name']
    
    st.write(f'Selected product name : {product_name}')
    st.write(f'Selected categorie : {categorie}')
    image = Image.open(choice_image)
    st.image(image)

    st.header('Visualisation of close images')
    # st.dataframe(embedding_data.head())
    # st.dataframe(embedding_data.loc[choice_image.stem])
    closest_to_query = cosines.loc[choice_image.stem].sort_values()
    # remove if categorie is different not similar
    closest_to_query.name = 'cosine_distance'
    closest_to_query = closest_to_query.reset_index()
    closest_to_query['categories'] = closest_to_query['image_name'].apply(lambda x: x.split('_')[0])
    closest_to_query = closest_to_query.set_index('categories').loc[categorie]
    # st.dataframe(closest_to_query)

    images = []
    captions = []
    for row in closest_to_query.itertuples():
        images.append(Image.open(f'data/img/{row.image_name}.jpg'))
        # product_name = product_data[row.Index]['product_name']
        captions.append(f'{row.cosine_distance}')
    st.image(images, captions)
    


