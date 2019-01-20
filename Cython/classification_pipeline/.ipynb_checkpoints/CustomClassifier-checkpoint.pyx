import numpy as np
from sklearn.metrics import classification_report
from sklearn.base import clone
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef class AreaRugClassifier(object):
    cdef public vector[string] class_labels
    cdef double predict_threshold
    cdef int title_num_features
    cdef object base_estimator
    cdef public object model1, model2
    
    def __cinit__(self, object base_estimator, int tn, double predict_threshold=0.5):
        self.base_estimator = base_estimator
        self.predict_threshold = predict_threshold
        self.title_num_features = tn

    def fit(self, X, y):
        cdef vector[string] labels = y
        
        X1, X2 = X[:,:self.title_num_features], X[:,self.title_num_features:]
        
        self.model1 = clone(self.base_estimator)
        self.model1.fit(X1, labels)
        
        self.model2 = clone(self.base_estimator)
        self.model2.fit(X2, labels)
        
        self.class_labels = self.model1.classes_
        
    def predict_proba(self, X):
        X1, X2 = X[:,:self.title_num_features], X[:,self.title_num_features:]
        
        cdef double[:,:] preds_probs1 = self.model1.predict_proba(X1)
        cdef double[:,:] preds_probs2 = self.model2.predict_proba(X2)
        
        return np.maximum.reduce([preds_probs1, preds_probs2])

    def predict(self, X):
        cdef int x
        cdef double[:,:] preds_probs = self.predict_proba(X)
        cdef int[:] indices = np.argmax(preds_probs, axis=1).astype(np.int32)
        
        cdef vector[string] out
        for x in indices:
            out.push_back(self.class_labels[x])
        
        return out

    def score(self, X, y):
        cdef vector[string] pred_labels = self.predict(X)
        return classification_report(y, pred_labels)