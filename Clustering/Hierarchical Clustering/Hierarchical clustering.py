import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate sample data
n_samples = 300
n_features = 2
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=4, random_state=42)

# Compute linkage matrix
Z = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Zoom in to show the last few merged clusters
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='lastp', p=12)
plt.title('Truncated Dendrogram (Last 12 Clusters)')
plt.xlabel('Clusters')
plt.ylabel('Distance')
plt.show()

# Plot a dendrogram with a different linkage method
Z_single = linkage(X, method='single')
plt.figure(figsize=(10, 5))
dendrogram(Z_single)
plt.title('Hierarchical Clustering Dendrogram (Single Linkage)')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
