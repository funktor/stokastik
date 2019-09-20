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
from .BaseClassifierCore import BaseClassifier


class RLSClassifier(BaseClassifier):
    def __init__(self, n_neighbors, kernel, lambda_k, lambda_u, constant_threshold=0.0, learn_threshold=False, 
                 thres_search_space_size=201, normalize_L=False, num_iterations=10000, learning_rate=0.001, use_gradient_descent=False):
        
        super(RLSClassifier, self).__init__(n_neighbors, kernel, lambda_k, lambda_u, constant_threshold, learn_threshold, 
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

        J = np.diag(np.concatenate([np.ones(l), np.zeros(u)]))
        
        if Y_in.ndim == 1:
            Y_in = np.expand_dims(Y_in, 1)
        
        if u > 0:
            Y_in = np.concatenate([Y_in, np.zeros((u, Y_in.shape[1]))])
            
        self.alpha = np.zeros((n, Y_in.shape[1]))
        
        print('Computing alpha')
        if self.use_gradient_descent:
            JK = J.dot(K)
            KLK = K.dot(L).dot(K)
            K_KLK = self.lambda_k * K + ((self.lambda_u * l) / (l + u) ** 2) * KLK

            momentum = np.zeros((n, Y_in.shape[1]))

            mean = np.zeros((n, Y_in.shape[1]))
            varu = np.zeros((n, Y_in.shape[1]))

            beta_1, beta_2, epsilon = 0.9, 0.999, 10**-7

            def grad(alpha):
                return -(1.0/l) * (Y_in-JK.dot(alpha)).T.dot(JK) + alpha.T.dot(K_KLK)

            for j in range(self.num_iterations):
                g = grad(self.alpha).T
                g[g > 10] = 10
                mean = beta_1*mean + (1-beta_1)*g
                varu = beta_2*varu + (1-beta_2)*np.power(g, 2)

                mean_h = mean / (1 - np.power(beta_1, j+1))
                varu_h = varu / (1 - np.power(beta_2, j+1))

                self.alpha -= self.learning_rate * mean_h/(np.sqrt(varu_h) + epsilon)
                
        else:
            print('Computing final matrix')
            final = (J.dot(K) + self.lambda_k * l * np.identity(l + u) + ((self.lambda_u * l) / (l + u) ** 2) * L.dot(K))
            #Compute Moore-Penrose PseudoInverse when data is large
            if final.shape[0] > 20000:
                final_inv = np.linalg.pinv(final)
            else:
                final_inv = np.linalg.inv(final)
            print('done')
            self.alpha = final_inv.dot(Y_in)
            
        print('done')
        
        if self.learn_threshold:
            self.thresholds = utils.compute_thresholds(K, self.alpha, Y_in, l, u, search_space_size=self.thres_search_space_size)
        else:
            self.thresholds = [self.constant_threshold]*Y_in.shape[1]