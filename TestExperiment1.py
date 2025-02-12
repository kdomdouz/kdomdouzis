# kmeans_clustering.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data (or replace with your own dataset)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit the KMeans model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Get the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the data and cluster centers
plt.figure(figsize=(8, 6))

# Plot the points with their assigned cluster color
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, marker='o')

# Plot the centroids
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

# Title and labels
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()
