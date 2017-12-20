from sklearn import naive_bayes as skbayes
from sklearn import datasets

from src.utils import dataset 
from src.utils import evaluation
from src.supervised_learning.naive_bayes import GaussianNaiveBayes

def main():
    # Load dataset
    data = datasets.load_digits()
    x = data.data
    y = data.target
    x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(x, y, split_size=0.6, shuffle=True)

    # GaussianNaiveBayes
    cls = GaussianNaiveBayes()
    cls.train(x_train, y_train)
    y_pred = cls.predict(x_test)
    print("Classification result: {}".format(evaluation.accuracy(y_pred, y_test)))

    # Scikit-learn GaussianNaiveBayes
    cls = skbayes.GaussianNB()
    y_pred = cls.fit(x_train, y_train).predict(x_test)
    print("Scikit-learn classification result: {}".format(evaluation.accuracy(y_pred, y_test)))

if __name__ == '__main__':
    main()
