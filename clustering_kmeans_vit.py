from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# read data
X = pd.read_csv('vit-base-patch16-224.csv', sep=',',index_col=0)

# apply model
kmeans = KMeans(n_clusters=2, random_state=42, verbose=1)
kmeans.fit(X)

kmeans.cluster_centers_# cluster centers result
y_pred = kmeans.labels_# labels result

# copy labels to CSV
X_to_csv = X.copy()
X_to_csv['label'] = y_pred
X_to_csv.to_csv('labels_vit_kmeans_2.csv')

# visualize results
tsne = TSNE() # reduction of dimensions
X_transformer = tsne.fit_transform(X)

X_transformer = pd.DataFrame(X_transformer, columns=['dim_0', 'dim_1'])
X_transformer['labels'] = y_pred

for i in range(2):
    selected = X_transformer[X_transformer['labels'] == i]
    x = selected['dim_0']
    y = selected['dim_1']
    plt.scatter(x, y, label=i, alpha=0.1)
plt.legend()
plt.show()