from sklearn.feature_extraction.text import TfidfVectorizer
import Utilities as utils
from gensim.models import Word2Vec, FastText
import numpy as np

def get_weighted_sentence_vectors(sentences, vector_model, idf_scores, vocabulary, vector_dim):
    tokenized_sentences = [utils.get_tokens(sent, min_ngram=1, max_ngram=1) for sent in sentences]
    docvecs = []
    
    for tokens in tokenized_sentences:
        vectors, weights = [], []
        for word in tokens:
            if word in vocabulary:
                weights.append(idf_scores[vocabulary[word]])
            else:
                weights.append(0.0)
                
            if word in vector_model.wv:
                vectors.append(vector_model.wv[word])
            else:
                vectors.append([0.0] * vector_dim)
                
        if np.sum(weights) == 0:
            prod = np.array([0.0] * vector_dim)
        else:
            prod = np.dot(weights, vectors) / np.sum(weights)
            
        docvecs.append(prod)
    return np.array(docvecs)

    
class W2VFeatures(object):
    def __init__(self, num_components=128, features=None, wv_type='W2V'):
        self.wv_model = None    
        self.feature_extractor = None
        self.num_components = num_components
        self.features = features
    
    def fit(self, X, y=None):
        if wv_type == 'W2V':
            self.wv_model = Word2Vec(alpha=0.025, size=self.num_components, window=5, min_alpha=0.025, 
                                     min_count=1, workers=10, negative=10, hs=0, iter=50)
        else:
            self.wv_model = FastText(size=self.num_components, window=5, min_count=1, workers=5, iter=50)
            
        self.feature_extractor = TfidfVectorizer(tokenizer=utils.get_tokens, ngram_range=(1, 1), stop_words='english', vocabulary=self.features)
        
        transformed_features = self.feature_extractor.fit_transform(X)
        tokenized_sentences = [utils.get_tokens(sent, min_ngram=1, max_ngram=1) for sent in X]
        
        self.wv_model.build_vocab(tokenized_sentences)
        self.wv_model.train(tokenized_sentences, total_examples=self.wv_model.corpus_count, epochs=50)
        
    def transform(self, X):
        return get_weighted_sentence_vectors(np.asarray(X), self.wv_model, self.feature_extractor.idf_, 
                                             self.feature_extractor.vocabulary_, self.num_components)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        
