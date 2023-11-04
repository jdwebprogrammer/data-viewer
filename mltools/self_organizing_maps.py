import numpy as np
import matplotlib.pyplot as plt
import argparse

class SelfOrganizingMap:
    def __init__(self, grid_size=(5, 5), input_dim=2, learning_rate=0.1, num_epochs=100):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)

    def find_best_matching_unit(self, input_vector):
        # Calculate the Euclidean distance between input_vector and all weights
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        # Find the indices of the neuron with the smallest distance (Best Matching Unit)
        bmu_indices = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_indices

    def update_weights(self, input_vector, bmu_indices, epoch):
        # Calculate the influence on each weight based on the epoch
        influence = np.exp(-epoch / self.num_epochs)
        # Update the weights of the neurons based on the input_vector and influence
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                weight_diff = input_vector - self.weights[i, j]
                self.weights[i, j] += self.learning_rate * influence * weight_diff

    def train(self, data):
        for epoch in range(self.num_epochs):
            for input_vector in data:
                bmu_indices = self.find_best_matching_unit(input_vector)
                self.update_weights(input_vector, bmu_indices, epoch)

    def plot(self, data):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], c='b', marker='o', label='Data Points')
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                plt.scatter(self.weights[i, j, 0], self.weights[i, j, 1], c='r', marker='x', s=100, label='SOM Neuron')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Self-Organizing Map (SOM)')
        plt.legend()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Self-Organizing Map (SOM)')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[5, 5], help='Grid size as two integers (default is [5, 5])')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension (default is 2)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate (default is 0.1)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs (default is 100)')
    args = parser.parse_args()

    data = np.random.rand(100, 2)  # Sample 2D data points
    som = SelfOrganizingMap(grid_size=args.grid_size, input_dim=args.input_dim, learning_rate=args.learning_rate, num_epochs=args.num_epochs)
    som.train(data)
    som.plot(data)

if __name__ == "__main__":
    main()
