import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class DimensionalityReductionPCA:
    def __init__(self, n_components=None):
        """
        Initialize the DimensionalityReductionPCA instance.

        Parameters:
        - n_components (int or None): The number of principal components to keep.
          If None (default), all components are kept.
        """
        self.n_components = n_components
        self.pca = None

    def fit_transform(self, data):
        """
        Fit the PCA model to the data and transform it to the reduced dimension.

        Parameters:
        - data (array-like): The data for dimensionality reduction.

        Returns:
        - array: The reduced-dimension data.
        """
        self.pca = PCA(n_components=self.n_components)
        reduced_data = self.pca.fit_transform(data)
        return reduced_data

    def explained_variance_ratio(self):
        """
        Get the explained variance ratio of the principal components.

        Returns:
        - array: The explained variance ratio of each principal component.
        """
        if self.pca is not None:
            return self.pca.explained_variance_ratio_
        else:
            raise ValueError("PCA model has not been fitted yet.")

    def plot_variance_explained(self):
        """
        Plot the cumulative explained variance ratio of the principal components.
        """
        if self.pca is not None:
            explained_variance_ratio = self.pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.title('Cumulative Explained Variance Ratio vs. Number of Principal Components')
            plt.grid()
            plt.show()
        else:
            raise ValueError("PCA model has not been fitted yet.")

def dimensionality_reduction_example():
    # Example usage of the DimensionalityReductionPCA class:
    data = np.random.rand(100, 3)  # Sample data with 3 features
    pca_reducer = DimensionalityReductionPCA(n_components=2)
    reduced_data = pca_reducer.fit_transform(data)
    pca_reducer.plot_variance_explained()