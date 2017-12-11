import numpy as np
import copy
import math
from collections import Counter

from src.utils import data
from src.utils import evaluation

# ===================
# K-NEAREST NEIGHBORS 
# ===================

class KNearestNeighbors(object):
    """
        K-Nearest Neighbors (KNN) is a non-parametric classsifier.
        Looks at the L points in the training set that are nearest to the test input x,
            count how many members of each class are in this set, and returns that empirical
            fraction as the estimate

        p(y = x|x, D, K) = 1/K * \sum(1 if yi = c else 0)
        
        Args:
            k (int): hyperparameter controlling the number of closest neighbors
    """
    def __init__(self, k=2):
        # "closest neighbors" when k > 1
        # "nearest neighbors" when k == 1
        self.k = k

    """
        Predict the targets based on the training, and test sets
        Args:
            x_train (float[][]): training set
            y_train (float[]|string[]): training target labels
            x_test (float[][]): test set
        Returns:
            float[]|string[]
    """
    def predict(self, x_train, y_train, x_test):
        # Hold the prediction
        y_pred = np.zeros(x_test.shape[0])
        K = min(self.k, x_train.shape[0])

        # Find the distances between the test set and train set
        distances = self._test_training_distance(x_test, x_train)

        # Find the closest distance for each test value
        for i in range(x_test.shape[0]):
            # Sort the distance from "close" to "far" and get the k neighbors
            sorted_distance = np.argsort(distances[i, :], axis=0)
            k_nearest_neighbors = y_train[sorted_distance[:K]]

            # Count the most frequent value
            labels, counts = np.unique(k_nearest_neighbors, return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]

        return y_pred

    """
        The distance between the test and training sample.
        l2 distance (euclidean distance) between the vectors.

        (x-y)^2 = x^2 + y^2 - 2xy

        Args:
            x (float[][]): x value in above equation, will be broadcasted to match y
            y (float[][]): y value in above equation
        Returns:
            float[][]
    """
    def _test_training_distance(self, x, y):
        x_square = np.sum(np.square(x), axis=1)
        y_square = np.sum(np.square(y), axis=1)
        x_y = np.dot(x, y.T)

        return np.sqrt(np.reshape(x_square, (-1, 1)) + y_square - 2.*x_y)

                
            
