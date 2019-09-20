from sklearn.base import BaseEstimator, ClassifierMixin
from .RLSClassifierCore import RLSClassifier
from .LapSVMClassifierCore import LapSVMClassifier
from .RLRClassifierCore import RLRClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import kneighbors_graph, BallTree
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier

class LaplacianMRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, algorithms=['RLS'], n_neighbors=4, kernel=RBF(1,(1,10)), lambda_k=0.1, lambda_u=0.01, 
                 u_split=0.8, threshold=0.0, strategy='IC'):
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        self.u_split = u_split
        self.threshold = threshold
        self.algorithms = algorithms
        self.num_classes = None
        self.strategy = strategy
        
        self.models = []
        
        if 'RLS' in algorithms:
            self.models += [RLSClassifier(self.n_neighbors, self.kernel, self.lambda_k, self.lambda_u, self.threshold, learn_threshold=False)]
        if 'LapSVM' in algorithms:
            self.models += [LapSVMClassifier(self.n_neighbors, self.kernel, self.lambda_k, self.lambda_u, self.threshold, learn_threshold=False)]
        if 'RLR' in algorithms:
            self.models += [RLRClassifier(self.n_neighbors, self.kernel, self.lambda_k, self.lambda_u, self.threshold, learn_threshold=False)]
        
        if self.strategy == 'OVA':
            for i in range(len(self.models)):
                self.models[i] = OneVsRestClassifier(self.models[i])
        
    def fit(self, X, y, X_unlabelled=None):
        y = np.array(y)
        self.num_classes = y.shape[1]
        
        if X_unlabelled is None:
            X_labelled_indices, X_unlabelled_indices = train_test_split(range(X.shape[0]), test_size=self.u_split, random_state=42)
            X_labelled, X_unlabelled = X[X_labelled_indices,:], X[X_unlabelled_indices,:]
            y_labelled = y[X_labelled_indices]
        else:
            X_labelled, y_labelled = X, y
            
        for i in range(len(self.models)):
            self.models[i].fit([X_labelled, X_unlabelled], y_labelled)
                
        return self
    
    def predict(self, X, y=None):
        preds = np.ones((X.shape[0], self.num_classes), dtype='int8')
        for i in range(len(self.models)):
            preds = preds & self.models[i].predict(X)
        return preds
    
    def score(self, X, y=None):
        preds = self.predict(X)
        p, r, f, s = precision_recall_fscore_support(y, preds)
        
        p = np.sum(p*s)/np.sum(s)
        r = np.sum(r*s)/np.sum(s)
        f = np.sum(f*s)/np.sum(s)
        
        return {'precision':p, 'recall':r, 'f1':f}
    
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
