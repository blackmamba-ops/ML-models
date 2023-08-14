import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Generate sample data
n_samples = 300
n_features = 2
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=4, random_state=42)

# Initialize Gaussian Mixture Model
gmm = GaussianMixture(n_components=4)

# Fit model to the data
gmm.fit(X)

# Visualize GMM clustering step by step
plt.figure(figsize=(12, 6))

for i in range(1, 5):
    plt.subplot(2, 2, i)

    # Fit model with fewer components for visualization
    gmm_step = GaussianMixture(n_components=i)
    gmm_step.fit(X)
    labels = gmm_step.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker='o', s=50)
    plt.title(f'Iteration {i}')

plt.tight_layout()
plt.show()
