from sklearn.feature_extraction.text import TfidfVectorizer
import Utilities as utils
from gensim.models import Word2Vec, FastText
import numpy as np
from multiprocessing import Pool

def get_weighted_sentence_vector(sentence, vector_model, idf_scores, vocabulary, vector_dim):
    vectors, weights = [], []
    for word in utils.get_tokens(sentence, min_ngram=1, max_ngram=1):
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
            
    return np.array(prod)

def sent_fn(x, wv_model, feature_extractor_idf, feature_extractor_vocab, num_components):
    return get_weighted_sentence_vector(x, wv_model, feature_extractor_idf, feature_extractor_vocab, num_components)

    
class W2VFeatures(object):
    def __init__(self, num_components=128, features=None, wv_type='W2V'):
        self.wv_model = None    
        self.feature_extractor = None
        self.num_components = num_components
        self.features = features
        self.wv_type = wv_type
    
    def fit(self, X, y=None):
        if self.wv_type == 'W2V':
            self.wv_model = Word2Vec(alpha=0.025, size=self.num_components, window=5, min_alpha=0.025, 
                                     min_count=10, workers=10, negative=5, hs=0, iter=5)
        else:
            self.wv_model = FastText(size=self.num_components, window=5, min_count=10, workers=5, iter=5)
            
        self.feature_extractor = TfidfVectorizer(tokenizer=utils.get_tokens, ngram_range=(1, 1), stop_words='english', vocabulary=self.features)
        
        transformed_features = self.feature_extractor.fit_transform(X)
        
        tokenized_sentences = []
        for sent in X:
            tokens = utils.get_tokens(sent)
            tokens = tokens[:min(len(tokens), 150)]
            tokenized_sentences.append(tokens)
        
        self.wv_model.build_vocab(tokenized_sentences)
        self.wv_model.train(tokenized_sentences, total_examples=self.wv_model.corpus_count, epochs=5)
        
    def parallel_pool(self, x):
        return get_weighted_sentence_vector(x, self.wv_model, self.feature_extractor.idf_, self.feature_extractor.vocabulary_, self.num_components)
        
    def transform(self, X):
        return [get_weighted_sentence_vector(x, self.wv_model, self.feature_extractor.idf_, 
                                             self.feature_extractor.vocabulary_, self.num_components) for x in np.asarray(X)]
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)