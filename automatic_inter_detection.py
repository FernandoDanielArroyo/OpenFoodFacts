
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import json

def get_cosine_distances(embedding_data):
    """
    index of embedding data is supposed to be image_name
    """
    cosines = cosine_distances(embedding_data)
    cosines = pd.DataFrame(cosines, index=embedding_data.index, columns=embedding_data.index)
    return cosines


def main():
    embedding = pd.read_csv('data/labels_vit_kmeans_2.csv', index_col=0).astype('float')
    embedding = embedding[embedding['label'] == 1].drop(columns='label')
    embedding.index.name = 'image_name'

    cosines = get_cosine_distances(embedding)
    cosines['category'] = cosines.index.to_series().apply(lambda x:x.split('_')[0])
    cosines = cosines.set_index('category', append=True)


    duplicates = {}
    for col in tqdm(cosines.columns, total=len(cosines.columns)):
        category = col.split('_')[0]
        df = cosines.loc[:, col]
        df.name = 'distance'
        df = df.reset_index(level=1)
        ordered = df[df['category'] != category].sort_values('distance')
        min_dist = ordered.iloc[0,1]
        target_img = ordered.index[0]
        duplicates[col] = (target_img, min_dist)

    with open('data/inter_product_duplicated.json', 'w') as f:
        json.dump(duplicates, f)


if __name__ == '__main__':
    main()
