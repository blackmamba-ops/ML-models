import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic gene expression data
random.seed(42)
np.random.seed(42)

# Number of genes and samples
num_genes = 500
num_samples = 100

# Create synthetic gene expression data
gene_data = np.random.rand(num_genes, num_samples) * 10

# Perform K-means clustering with n_clusters=3
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
labels = kmeans.fit_predict(gene_data.T)

# Plot gene expression profiles with different colors for each cluster
plt.figure(figsize=(12, 6))
for i in range(n_clusters):
    cluster_samples = gene_data[:, labels == i]
    plt.plot(cluster_samples, alpha=0.5, label=f"Cluster {i+1}")

plt.xlabel("Samples")
plt.ylabel("Gene Expression")
plt.title("Gene Expression Profiling using K-means Clustering")
plt.legend()
plt.grid()
plt.show()


