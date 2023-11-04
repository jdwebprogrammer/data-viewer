import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

class AnomalyDetectionIsolationForest:
    def __init__(self, contamination=0.05, random_state=None):
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = None

    def fit_predict(self, data):
        self.isolation_forest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        anomaly_scores = self.isolation_forest.fit_predict(data)
        return anomaly_scores

    def plot_anomalies(self, data):
        if self.isolation_forest is not None:
            anomaly_scores = self.isolation_forest.decision_function(data)
            plt.scatter(data[:, 0], data[:, 1], c=anomaly_scores, cmap='viridis')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Anomaly Detection using Isolation Forest')
            plt.colorbar(label='Anomaly Score')
            plt.show()
        else:
            raise ValueError("Isolation Forest model has not been fitted yet.")

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection using Isolation Forest')
    parser.add_argument('--contamination', type=float, default=0.05, help='Expected proportion of outliers in the data (default is 0.05)')
    parser.add_argument('--random_state', type=int, default=None, help='Seed for random number generation (default is None)')
    args = parser.parse_args()

    # Example usage of the AnomalyDetectionIsolationForest class:
    data = np.random.randn(200, 2)  # Sample data with 2 features
    anomaly_detector = AnomalyDetectionIsolationForest(contamination=args.contamination, random_state=args.random_state)
    anomaly_scores = anomaly_detector.fit_predict(data)
    anomaly_detector.plot_anomalies(data)

if __name__ == "__main__":
    main()