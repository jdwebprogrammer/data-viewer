import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

class DensityEstimationKDE:
    def __init__(self, bandwidth=1.0):
        """
        Initialize the DensityEstimationKDE instance.

        Parameters:
        - bandwidth (float): Bandwidth parameter for the kernel density estimation (default is 1.0).
        """
        self.bandwidth = bandwidth
        self.data = None
        self.kde = None

    def fit(self, data):
        """
        Fit the KDE model to the data.

        Parameters:
        - data (array-like): The data for density estimation.
        """
        self.data = data
        self.kde = norm(loc=np.mean(data), scale=np.std(data) * self.bandwidth)

    def pdf(self, x):
        """
        Get the probability density function (PDF) values for given input values.

        Parameters:
        - x (array-like): Input values for which to compute the PDF.

        Returns:
        - array: PDF values corresponding to the input values.
        """
        if self.kde is not None:
            return self.kde.pdf(x)
        else:
            raise ValueError("KDE model has not been fitted yet.")

    def plot_density(self, xmin=None, xmax=None, num_points=1000):
        """
        Plot the estimated probability density function (PDF).

        Parameters:
        - xmin (float): Minimum x-value for the plot (default is None).
        - xmax (float): Maximum x-value for the plot (default is None).
        - num_points (int): Number of points to generate for the plot (default is 1000).
        """
        if self.kde is not None:
            if xmin is None:
                xmin = min(self.data)
            if xmax is None:
                xmax = max(self.data)
            
            x_values = np.linspace(xmin, xmax, num_points)
            pdf_values = self.pdf(x_values)
            
            plt.plot(x_values, pdf_values, label='PDF')
            plt.xlabel('X')
            plt.ylabel('PDF')
            plt.title('Kernel Density Estimation (KDE) Plot')
            plt.legend()
            plt.show()
        else:
            raise ValueError("KDE model has not been fitted yet.")

def main():
    parser = argparse.ArgumentParser(description='Kernel Density Estimation (KDE) Plot')
    parser.add_argument('--bandwidth', type=float, default=1.0, help='Bandwidth parameter for KDE (default is 1.0)')
    parser.add_argument('--data_file', type=str, default=None, help='File path containing data for density estimation')
    parser.add_argument('--xmin', type=float, default=None, help='Minimum x-value for the plot (default is None)')
    parser.add_argument('--xmax', type=float, default=None, help='Maximum x-value for the plot (default is None)')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of points for the plot (default is 1000)')
    args = parser.parse_args()

    # Load data from a file if specified, otherwise generate random data
    if args.data_file:
        data = np.loadtxt(args.data_file)
    else:
        data = np.random.randn(100)  # Sample data from a normal distribution

    kde_estimator = DensityEstimationKDE(bandwidth=args.bandwidth)
    kde_estimator.fit(data)
    kde_estimator.plot_density(xmin=args.xmin, xmax=args.xmax, num_points=args.num_points)

if __name__ == "__main__":
    main()