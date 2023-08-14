import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Create a toy dataset
X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)

# Initialize DBSCAN with parameters
eps = 0.3
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit and predict clusters
labels = dbscan.fit_predict(X)


# Visualize the results
def plot_clusters(X, labels, title):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    markers = ['o', 's', '^', 'D', 'P', '*', 'X']
    colors = plt.cm.get_cmap('tab20', n_clusters)

    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    marker=markers[i % len(markers)], color=colors(i),
                    label=f'Cluster {label if label != -1 else "Noise"}')

    plt.title(title)
    plt.legend()
    plt.show()


# Initial plot (before clustering)
plot_clusters(X, np.zeros(X.shape[0]), title="Before Clustering")

# Step 1: After the first round of clustering
plot_clusters(X, labels, title="Step 1: Clustering with DBSCAN")

# Step 2: After modifying epsilon and re-clustering
eps = 0.2
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X)
plot_clusters(X, labels, title="Step 2: Re-clustering with Modified Epsilon")

# Step 3: After modifying min_samples and re-clustering
min_samples = 10
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X)
plot_clusters(X, labels, title="Step 3: Re-clustering with Modified Min Samples")
