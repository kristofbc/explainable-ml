import numpy as np

from src.utils import data
from src.utils import evaluation

# ======================
# NAIVE BAYES CLASSIFIER
# ======================

class GaussianNaiveBayes(object):
    """
        Conditional distribution assuming conditionally independent features given the class label.
        "naive" because we do not expect the features to be independent, even conditional on the class label

        P(y|x) = \dfrac{P(x|y)P(y)}{P(x)} == p(class|data) = \dfrac{p(data|class)P(class)}{p(data)}

        y = class, x = data
        p(y|x) = posterior
        p(x|y) = likelihood
        p(y) = prior
        p(x) = marginal probability

        "naive" assumption == independence : p(x1, x2, x3|y) = p(x1|y)*p(x2|y)*p(x3|y)
            p(x*|y) is computed via the probability density function of the normal distribution
            e.g., p(x1|y) = \dfrac{1}{\sqrt{2*\pi*variance of x[y == Y]}} * \exp{\dfrac{-(x1-\overbar{y})^2}{2*variance of x[y == Y]}}

        Args:
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.means = None
        self.variances = None
        self.priors = None

    """
        Train the classifier: get the mean and the variance of each feature
        Args:
            x (float[][]): training dataset
            y (float[]): labels for the training dataset
    """
    def train(self, x, y):
        # Required parameters
        self.x = x
        self.y = y

        classes = np.unique(y)
        n_classes = len(classes)
        n_features = x.shape[1]
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        # For each feature, get the mean and variance corresponding to the class
        # Fetch also the prior of the class
        for i in range(n_classes):
            self.priors[i] = self.prior(classes[i])
            x_y_class = x[np.where(y == classes[i])]
            # Extract the feature corresponding to the selected data matching the class
            for j in range(n_features):
                self.means[i][j] = np.mean(x_y_class[:, j])
                self.variances[i][j] = np.var(x_y_class[:, j])

    """
        Apply the classifier to new data points
        We ignore the marginal probability (denominator)

        "naive" assumption == independence : p(x1, x2, x3|y) = p(x1|y)*p(x2|y)*p(x3|y)

        Args:
            x (float[][]): sample to classify
        Returns:
            float[]
    """
    def predict(self, x):
        # Parameters of the prediction
        classes = np.unique(self.y)
        n_classes = len(classes)
        n_samples = x.shape[0]
        n_features = x.shape[1]
        y_pred = np.zeros(n_samples)

        # Loop through each samples to compute the highest class score
        for i in range(n_samples):
            class_score = np.zeros(n_classes)
            # For each class, find the one with the highest score
            class_likelihoods = self.likelihood(x[i], self.means, self.variances)
            class_score = self.priors * np.prod(class_likelihoods, axis=1)

            # Below is an example with 1 or 2 more loop, the code below does exactly the same (use numpy broadcasting ability)
            # For each class, find the one with the highest score
            #for j in range(n_classes):
            #    prior = self.priors[j]
            #    likelihoods = self.likelihood(x[i], self.means[j], self.variances[j])
            #    # Evaluate each feature
            #    class_score[j] = prior * np.prod(likelihoods)

            #    # One more loop for the features
            #    #for k in range(n_features):
            #    #    x_value = x[i][k]
            #    #    likelihood = self.likelihood(x_value, self.means[j][k], self.variances[j][k])
            #    #    prior *= likelihood

            #    #class_score[j] = prior

            # Keep the highest score
            y_pred[i] = np.argmax(class_score)

        return y_pred


    """
        Compute the likelihood p(x|y) using the pdf of the Gaussian distribution

        p(x1|y) = \dfrac{1}{\sqrt{2*\pi*variance of x[y == Y]}} * \exp{\dfrac{-(x1-\overbar{y})^2}{2*variance of x[y == Y]}}

        Args:
            x (float): value of the data to classify
            y_mean (float): mean of the feature
            y_var (float): variance of the feature
        Returns:
            float
    """
    def likelihood(self, x, y_mean, y_var):
        eps = 1e-4 # Ensure no division by zero
        return (1.0 / np.sqrt(2.0 * np.pi * y_var + eps)) * \
               (np.exp(-(np.power(x - y_mean, 2.0) / (2.0 * y_var + eps))))

    """
        Compute the prior p(z): number of time z is found in the samples
        Args:
            x (float): find the occurence of this inputs in the samples
        Returns:
            float
    """
    def prior(self, z):
        match = self.x[np.where(self.y == z)]
        return len(match) / len(self.x)


class BayesianNaiveBayes(object):
    """
        Bayesian Naive Bayes (or Dirichlet-Multinomial Classifiers) assume the the features are 
            conditionnaly independent given the class label.
        "naive" because we do not expect the features to be independent, even conditional on class label.
        
        To overcome overfitting, this classifier is Bayesian: it use a factored prior at training time

        Args:
            alpha (float): alpha used in the Dirichlet distribution
            beta1 (float): beta1 used in the beta distribution
            beta2 (float): beta2 used in the beta distrbution
    """
    def __init__(self, alpha=1, beta1=1, beta2=1):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

    

from sklearn import naive_bayes as skbayes
from sklearn import datasets
from src.utils import dataset 
from src.utils import evaluation

data = datasets.load_digits()
x = data.data
y = data.target
x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(x, y, split_size=0.6, shuffle=True)

cls = GaussianNaiveBayes()
cls.train(x_train, y_train)
y_pred = cls.predict(x_test)
print("Classification result: {}".format(evaluation.accuracy(y_pred, y_test)))

cls = skbayes.GaussianNB()
y_pred = cls.fit(x_train, y_train).predict(x_test)
print("Scikit-learn classification result: {}".format(evaluation.accuracy(y_pred, y_test)))


