import numpy as np
import math

"""
    Split the data into two sets based on the split_size parameter
    Args:
        data (mixed[][]): data to split into two sets
        split_size (int|float): size of the split. int type indicate a position to split
            and float type indicate a size for the first split
        shuffle (boolean): if the dataset should be shuffled
    Returns:
        mixed[][], mixed[][]
"""
def split_dataset(data, split_size, shuffle=False):
    idx = np.arange(len(data))
    # Shuffle the order of indexes if required
    if shuffle:
        np.random.shuffle(idx)
    
    dataset = data[idx]
    # Split the data based on the int or float split_size
    if isinstance(split_size, float):
        split_idx = int(math.ceil(len(data)*split_size))
        return dataset[:split_idx], dataset[split_idx:]
    else:
        return dataset[:split_size], dataset[split_size:]

"""
    Split data to a training and test set
    Args:
        x (mixed[][]): training set to split
        y (mixed[]): target set to split
        split_size (int|float): size of the split. int type indicate a position to split
            and float type indicate a size for the first split
        shuffle (boolean): if the dataset should be shuffled
    Returns:
        x_train (mixed[][]), x_test (mixed[][]), y_train (mixed[][]), y_test (mixed[][])
"""
def split_train_test_dataset(x, y, split_size, shuffle=False):
    n_features = x.shape[1]
    # Concatenate training and targets for splitting the dataset
    x_y = np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)
    x_y_train, x_y_test = split_dataset(x_y, split_size, shuffle)
    # Recover the training and targets from the split dataset
    return x_y_train[:, :n_features], np.squeeze(x_y_train[:, n_features:]), x_y_test[:, :n_features], np.squeeze(x_y_test[:, n_features:])





