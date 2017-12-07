import numpy as np

"""1
    Test the accuracy between the predicted set and the true set
    Args:
        y_pred (float[]|string[]): prediction to evaluate
        y_true (float[]|string[]): true set used for evaluation
    Returns:
        float
"""
def accuracy(y_pred, y_true):
    return len(np.where(y_pred == y_true)[0]) / len(y_true)
