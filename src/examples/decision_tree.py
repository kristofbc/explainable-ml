from sklearn import datasets

from src.utils import dataset
from src.supervised_learning.decision_tree import DecisionTree

# To remove
def gain(y, y1, y2):
    print(y.shape, y1.shape, y2.shape)
    exit()

def leaf(y):
    print(y.shape)
    exit()

def worth_splitting(current_depth, largest_cost, largest_cost_set):
    print(current_depth, largest_cost, largest_cost_set)
    exit()
    
tree = DecisionTree(gain_function=gain, leaf_function=leaf, worth_splitting_function=worth_splitting)

# Dataset
def main():
    data = datasets.load_iris()
    x = data.data
    y = data.target
    x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(x, y, 0.90)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

if __name__ == '__main__':
    main()
