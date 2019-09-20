import numpy as np
import math
from numpy import linalg

import sklearn
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph, BallTree

import scipy.optimize as sco
import scipy.sparse as sp
import scipy as sc
from numpy.random import choice

from itertools import cycle, islice
from sklearn.gaussian_process.kernels import RBF
import cvxopt
from sklearn.metrics import precision_recall_fscore_support, classification_report
import classifiers.ClassifierUtils as utils
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors, kernel, lambda_k, lambda_u, constant_threshold=0.0, learn_threshold=False, 
                 thres_search_space_size=201, normalize_L=False, num_iterations=100, learning_rate=0.01, use_gradient_descent=False):
        
        self.n_neighbors = n_neighbors
        self.kernel =kernel

        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        
        self.X = None
        self.constant_threshold = constant_threshold
        self.learn_threshold = learn_threshold
        self.thres_search_space_size = thres_search_space_size
        self.normalize_L = normalize_L
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.use_gradient_descent = use_gradient_descent
        
        
    def fit(self, X, X_no_label, Y_in, use_preds=None):
        pass
    
    def predict(self, X):
        K = self.kernel(X, self.X)
        preds = K.dot(self.alpha)
        predictions = np.array((preds > self.thresholds) * 1)
        return np.array(predictions, dtype='int8')
    
    def predict_proba(self, X):
        K = self.kernel(X, self.X)
        return np.array(K.dot(self.alpha))
    
    def decision_function(self, X):
        K = self.kernel(X, self.X)
        return np.array(K.dot(self.alpha))
    
    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.sum(np.all(preds == y, axis=1))/float(y.shape[0])
    
    def metrics(self, X, y, class_names):
        preds = self.predict(X)
        print(classification_report(y, preds, target_names=class_names))
        a = precision_recall_fscore_support(y, preds, average='weighted')
        none_index = np.where((class_names == 'None') | (class_names == 'none'))[0]
        
        b = None
        if len(none_index) > 0:
            none_index = none_index[0]
            labels = list(range(none_index)) + list(range(none_index+1, len(class_names)))
            new_class_names = class_names[labels]
            print(classification_report(y, preds, labels=labels, target_names=new_class_names))
            b = precision_recall_fscore_support(y, preds, labels=labels, average='weighted')
        
        return a, b