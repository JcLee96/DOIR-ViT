# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# # %matplotlib inline
#
# X= -2 * np.random.rand(100,2)
# X1 = 1 + 2 * np.random.rand(50,2)
# X[50:100, :] = X1
# plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
# plt.show()
#
#
# from sklearn.cluster import KMeans
# Kmean = KMeans(n_clusters=2)
# Kmean.fit(X)
#
# Kmean.cluster_centers_

import torch
import numpy as np
from kmeans_pytorch import kmeans

# data
data_size, dims, num_clusters = 1000, 2, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)

# kmeans
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
)

print('a')