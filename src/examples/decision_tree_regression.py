import numpy as np
from sklearn import datasets
from sklearn import tree as sktree

from src.utils import dataset 
from src.utils import data
from src.utils import evaluation
from src.supervised_learning.decision_tree import RegressionDecisionTree

def main():
    # Load dataset, from http://scikit-learn.org/stable/modules/tree.html#regression
    x_train = [[0., 0.], [2., 2.]]
    y_train = [0.5, 2.5]
    x_test = [[1., 1.]]
    y_test = [0.5]

    # RegressionDecisionTree
    tree = RegressionDecisionTree(minimum_sample_count=1)
    tree.train(x_train, y_train)
    y_pred = tree.predict(x_test)
    print("Regression result: {}".format(evaluation.accuracy(y_pred, y_test)))

    # Sklearn tree
    clf = sktree.DecisionTreeRegressor()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Scikit-learn Regression result: {}".format(evaluation.accuracy(y_pred, y_test)))


if __name__ == '__main__':
    main()
