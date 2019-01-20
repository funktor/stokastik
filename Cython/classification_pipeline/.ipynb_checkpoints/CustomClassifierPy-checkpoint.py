from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import classification_report
from sklearn.base import clone

class AreaRugClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, tn, predict_threshold=0.5):
        self.base_estimator = base_estimator
        self.model1, self.model2 = None, None
        self.class_labels = None
        self.predict_threshold = predict_threshold
        self.title_num_features = tn

    def fit(self, X, y):
        X1, X2 = X.tocsc()[:,:self.title_num_features], X.tocsc()[:,self.title_num_features:]
        
        self.model1 = clone(self.base_estimator)
        self.model1.fit(X1, y)
        
        self.model2 = clone(self.base_estimator)
        self.model2.fit(X2, y)
        
        self.class_labels = self.model1.classes_
        
    def predict_proba(self, X):
        X1, X2 = X.tocsc()[:,:self.title_num_features], X.tocsc()[:,self.title_num_features:]
        
        preds_probs1 = self.model1.predict_proba(X1)
        preds_probs2 = self.model2.predict_proba(X2)
        
        return np.maximum.reduce([preds_probs1, preds_probs2])

    def predict(self, X):
        preds_probs = self.predict_proba(X)
        indices = np.argmax(preds_probs, axis=1)
        
        return self.class_labels[indices]

    def score(self, X, y):
        pred_labels = self.predict(X)
        return classification_report(y, pred_labels)