from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import Utilities as utils

class LDAFeatures(object):
    def __init__(self, num_components=128, features=None):
        self.lda = None
        self.feature_extractor = None
        self.num_components = num_components
        self.features = features
    
    def fit(self, X, y=None):
        self.lda = LatentDirichletAllocation(n_components=self.num_components, max_iter=50, random_state=0)
        self.feature_extractor = TfidfVectorizer(tokenizer=utils.get_tokens, ngram_range=(1, 1), stop_words='english', vocabulary=self.features)
        
        transformed_features = self.feature_extractor.fit_transform(X)
        self.lda.fit(transformed_features)
        
    def transform(self, X):
        transformed_features = self.feature_extractor.transform(X)
        return self.lda.transform(transformed_features)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        
