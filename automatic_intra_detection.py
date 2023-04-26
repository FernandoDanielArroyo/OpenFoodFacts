import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import argparse

def get_cosine_distances(embedding_data):
    """
    index of embedding data is supposed to be image_name
    """
    cosines = cosine_distances(embedding_data)
    cosines = pd.DataFrame(cosines, index=embedding_data.index, columns=embedding_data.index)
    return cosines

def print_statistics(data, categories):
    """
    print the statistics of the dictionnary
    """
    number_of_categories_with_duplications = len(data)
    percentage_of_categories_with_duplications = len(data)*100/len(categories)
    
    nb_duplicates = []
    for v in data.values():
        nb_duplicates.append(len(v))
    average_number_of_duplicates = np.mean(nb_duplicates)
    max_number_of_duplicates = np.max(nb_duplicates)

    print(f'{number_of_categories_with_duplications=}')
    print(f'{percentage_of_categories_with_duplications=:.3f}')
    print(f'{average_number_of_duplicates=:.2f}')
    print(f'{max_number_of_duplicates=}')

def main():

    THRESHOLD = 0.3 
    embedding = pd.read_csv('data/vit-base-patch16-224.csv', index_col=0)

    embedding.index.name = 'image_name'
    embedding = embedding.reset_index()
    embedding['categories'] = embedding['image_name'].apply(lambda x: x.split('_')[0])
    embedding = embedding.set_index('categories')
    categories = sorted(embedding.index.unique())
    duplicated_images_by_category = {}
    for i, category in tqdm(enumerate(categories), total=len(categories)):
        all_images_of_category = embedding.loc[category]
        if type(all_images_of_category) == pd.DataFrame:
            all_images_of_category = all_images_of_category.set_index('image_name')
            distances = get_cosine_distances(all_images_of_category)
            distances.index.name = None
            distances = distances.where(np.triu(np.ones(distances.shape)).astype(bool))
            distances = distances.stack()
            distances = distances[(distances > 0) & (distances < THRESHOLD)]
            pair_duplicates = []
            if not distances.empty:
                for index, distance in distances.items():
                    pair_duplicates.append((index[0], index[1], distance))
            if pair_duplicates:
                duplicated_images_by_category[category] = pair_duplicates
    print_statistics(duplicated_images_by_category, categories)
    with open('data/automated_duplicated.json', 'w') as out:
        json.dump(duplicated_images_by_category, out)

    
if __name__ == '__main__':
    main()

