import numpy as np
from sklearn import datasets
from sklearn import tree as sktree

from src.utils import dataset 
from src.utils import data
from src.utils import evaluation
from src.supervised_learning.decision_tree import ClassificationDecisionTree

def main():
    # Load dataset
    data = datasets.load_iris()
    x = data.data
    y = data.target
    x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(x, y, split_size=0.6, shuffle=True)

    # ClassificationDecisionTree
    tree = ClassificationDecisionTree()
    tree.train(x_train, y_train)
    pruned_tree = tree.prune()
    y_pred = tree.predict(x_test)
    y_pred_pruned = pruned_tree.predict(x_test)
    print("Classification result: {}".format(evaluation.accuracy(y_pred, y_test)))
    print("Classification result pruned: {}".format(evaluation.accuracy(y_pred_pruned, y_test)))

    # Sklearn tree
    clf = sktree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Scikit-learn classification result: {}".format(evaluation.accuracy(y_pred, y_test)))


if __name__ == '__main__':
    main()
