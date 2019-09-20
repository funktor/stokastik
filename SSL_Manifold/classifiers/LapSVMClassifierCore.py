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
from scipy.optimize import Bounds
from sklearn.metrics import precision_recall_fscore_support, classification_report
import classifiers.ClassifierUtils as utils
from .BaseClassifierCore import BaseClassifier


class LapSVMClassifier(BaseClassifier):
    def __init__(self, n_neighbors, kernel, lambda_k, lambda_u, constant_threshold=0.0, learn_threshold=False, 
                 thres_search_space_size=201, normalize_L=False, num_iterations=5000, learning_rate=0.001, use_gradient_descent=False):
        
        super(LapSVMClassifier, self).__init__(n_neighbors, kernel, lambda_k, lambda_u, constant_threshold, learn_threshold, 
                 thres_search_space_size, normalize_L, num_iterations, learning_rate, use_gradient_descent)
        
        
    def fit(self, X_in, Y_in):
        X, X_no_label = X_in
        
        Y_in[Y_in == 0] = -1
        
        if X_no_label is not None and X_no_label.shape[0] > 0:
            X_all = np.vstack((X, X_no_label))
            l, u = X.shape[0], X_no_label.shape[0]
            
        else:
            X_all = X
            l, u = X.shape[0], 0
        
        n = l + u
        self.X = X_all
        
        K, L = utils.compute_KL_matrix(X_all, self.n_neighbors, self.kernel, l, u, self.normalize_L)

        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)
        
        if Y_in.ndim == 1:
            Y_in = np.expand_dims(Y_in, 1)
        
        self.alpha = np.zeros((n, Y_in.shape[1]))
        
        for i in range(Y_in.shape[1]):
            Y_in_diag = np.diag(Y_in[:,i])
        
            print('Inverting matrix')
            if n > 20000:
                almost_alpha = np.linalg.pinv(2 * self.lambda_k * np.identity(l + u) + ((2 * self.lambda_u) / (l + u) ** 2) * L.dot(K)).dot(J.T).dot(Y_in_diag)
            else:
                almost_alpha = np.linalg.inv(2 * self.lambda_k * np.identity(l + u) + ((2 * self.lambda_u) / (l + u) ** 2) * L.dot(K)).dot(J.T).dot(Y_in_diag)

            print('Computing Q matrix')
            Q = Y_in_diag.dot(J).dot(K).dot(almost_alpha)
            print('done')
            
            print('Solving for beta')
            
            cons = {'type': 'eq', 'fun': lambda x: x.dot(Y_in[:,i]), 'jac': Y_in[:,i]}
            one_vec = np.ones(l)
            beta_hat = sco.minimize(lambda x: 0.5 * x.dot(Q).dot(x) - one_vec.dot(x), 
                                    np.zeros(l), 
                                    constraints=cons, 
                                    jac=lambda x: x.T.dot(Q) - one_vec,
                                    bounds=[(0.0, 1.0/l) for _ in range(l)], 
                                    method='L-BFGS-B')['x']
            print('done')

            print('Computing alpha')
            self.alpha[:,i] = almost_alpha.dot(beta_hat)
            print('done')
            
        
        if self.learn_threshold:
            self.thresholds = utils.compute_thresholds(K, self.alpha, Y_in, l, u, search_space_size=self.thres_search_space_size)
        else:
            self.thresholds = [self.constant_threshold]*Y_in.shape[1]