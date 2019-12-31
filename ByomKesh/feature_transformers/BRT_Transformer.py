from sklearn.feature_extraction.text import TfidfVectorizer
import Utilities as utils
from gensim.models import Word2Vec, FastText
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


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
                
            if word in vector_model:
                vectors.append(vector_model[word])
            else:
                vectors.append([0.0] * vector_dim)
                
        if np.sum(weights) == 0 or np.sum(vectors) == 0:
            prod = np.array([0.0] * vector_dim)
        else:
            prod = np.dot(weights, vectors) / np.sum(weights)
        docvecs.append(prod)
    return np.array(docvecs)


def load_glove_model(glove_path):
    print("Loading Glove Model")
    f = open(glove_path,'r',encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

    
class BRTFeatures(object):
    def __init__(self, num_components=768, features=None, bert_path='glove.6B.300d.txt'):
        self.wv_model = None    
        self.feature_extractor = None
        self.num_components = num_components
        self.features = features
        self.glove_path = glove_path
    
    def fit(self, X, y=None):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        marked_sents = ["[CLS] " + text + " [SEP]" for text in X]
        tokenized_texts = [tokenizer.tokenize(text) for text in marked_sents]
        
        self.glove_model = load_glove_model(self.glove_path)
        self.feature_extractor = TfidfVectorizer(tokenizer=utils.get_tokens, ngram_range=(1, 1), stop_words='english', vocabulary=self.features)
        self.feature_extractor.fit(X)
        
    def transform(self, X):
        return get_weighted_sentence_vectors(np.asarray(X), self.glove_model, self.feature_extractor.idf_, 
                                             self.feature_extractor.vocabulary_, self.num_components)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        
