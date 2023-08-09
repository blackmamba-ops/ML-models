import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Synthetic gene expression data
# Replace this with your actual gene expression dataset
num_samples = 100      #rows
num_genes = 500        #features
np.random.seed(42)
gene_expression = np.random.rand(num_samples, num_genes)

# Preprocessing: Standardize the data
scaler = StandardScaler()
scaled_gene_expression = scaler.fit_transform(gene_expression)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_gene_expression)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA for Gene Expression Analysis')
plt.grid()
plt.show()



