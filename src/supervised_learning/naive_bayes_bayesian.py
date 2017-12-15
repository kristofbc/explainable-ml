import numpy as np

from src.utils import data
from src.utils import evaluation

# ===============================
# BAYESIAN NAIVE BAYES CLASSIFIER
# ==============================

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

    

