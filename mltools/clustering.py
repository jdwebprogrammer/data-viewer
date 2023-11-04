import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import argparse

class KMeansClustering:
    def __init__(self, n_clusters=3, random_state=0):
        """
        Initialize the KMeansClustering instance.

        Parameters:
        - n_clusters (int): The number of clusters to create (default is 3).
        - random_state (int): Seed for random number generation (default is 0).
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None

    def fit(self, data):
        """
        Fit the K-Means model to the data.

        Parameters:
        - data (array-like): The data to cluster.
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(data)

    def get_cluster_assignments(self):
        """
        Get cluster assignments for each data point.

        Returns:
        - array: An array of cluster assignments.
        """
        if self.kmeans is not None:
            return self.kmeans.labels_
        else:
            raise ValueError("K-Means model has not been fitted yet.")

    def get_cluster_centers(self):
        """
        Get the cluster centers.

        Returns:
        - array: An array of cluster centers.
        """
        if self.kmeans is not None:
            return self.kmeans.cluster_centers_
        else:
            raise ValueError("K-Means model has not been fitted yet.")

    def plot_clusters(self, data):
        """
        Visualize the clusters.

        Parameters:
        - data (array-like): The data used for clustering.
        """
        if self.kmeans is not None:
            cluster_assignments = self.kmeans.labels_
            cluster_centers = self.kmeans.cluster_centers_

            plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='viridis')
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='X', label='Centroids')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('K-Means Clustering')
            plt.legend()
            plt.show()
        else:
            raise ValueError("K-Means model has not been fitted yet.")

def main():
    parser = argparse.ArgumentParser(description='K-Means Clustering')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters to create (default is 3)')
    parser.add_argument('--random_state', type=int, default=0, help='Seed for random number generation (default is 0)')
    args = parser.parse_args()

    # Example usage of the KMeansClustering class:
    data, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    kmeans_clusterer = KMeansClustering(n_clusters=args.n_clusters, random_state=args.random_state)
    kmeans_clusterer.fit(data)
    kmeans_clusterer.plot_clusters(data)

if __name__ == "__main__":
    main()