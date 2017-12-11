from sklearn import datasets
from sklearn import neighbors as skneighbors

from src.utils import dataset 
from src.utils import data
from src.utils import evaluation
from src.supervised_learning.k_nearest_neighbors import KNearestNeighbors

def main():    
    # Load dataset
    data = datasets.load_iris()
    x = data.data
    y = data.target
    x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(x, y, split_size=0.6, shuffle=True)

    clf = KNearestNeighbors()
    y_pred = clf.predict(x_train, y_train, x_test)
    print("Classification result: {}".format(evaluation.accuracy(y_pred, y_test)))

    # Sklearn KNeighborsClassifier
    clf = skneighbors.KNeighborsClassifier(n_neighbors=2)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Scikit-learn classification result: {}".format(evaluation.accuracy(y_pred, y_test)))

if __name__ == '__main__':
    main()
