import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, AgglomerativeClustering

wine = datasets.load_wine()
X = wine.data
# Z = linkage(X, 'single')
# plt.figure(figsize = (25,10))
# plt.title('Dendrogram: Linkage')
# dendrogram(Z, truncate_mode="level", p =10,show_leaf_counts=True)
# dendrogram(Z)
# kclus = fcluster(Z, 3, criterion='maxclust')
# print(kclus)
data = pd.DataFrame(data=wine['data'], columns=wine['feature_names'])
data1 = data.drop(["alcohol", "ash", "hue"], axis='columns')
data2 = data.drop(["alcohol", "ash", "hue", "magnesium", "flavanoids"], axis='columns')
single2 = linkage(data1, method='single', metric='euclidean')
# plt.figure(figsize=(25, 10))
# dendrogram(single2, truncate_mode="level", p=10, show_leaf_counts=True)
# plt.title('Dendrogram: Average Linkage')
# plt.show()
# # fig.savefig('dendrogram.png')

name = 'complete'
clusters = 6
model = AgglomerativeClustering(linkage=name,affinity="euclidean", n_clusters=clusters, compute_full_tree=True, )
model = model.fit(data2)
fig = plt.figure(figsize=[10,7])
plt.scatter(X[:,0],X[:,12],c=model.labels_,cmap='rainbow_r')
plt.title('Agglomerative Clustering: '+name + ", " + str(clusters) + " clusters")
plt.show()
# fig.savefig(name+ ' ' + str(clusters) + " Aglo graph.png")

