import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import kneighbors_graph, BallTree
import scipy.sparse as sp
import scipy as sc
import pickle, os, re, numpy as np, gensim, time, sys
import pandas as pd, math, collections, random, tables
import numpy as np
import pandas as pd
import json
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_fscore_support
from scipy import sparse
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import defaultdict
import csv
import os, re, math, random

def ngrams(tokens, gram_len=2):
    out = []
    for i in range(len(tokens)-gram_len+1):
        new_token = ' '.join(tokens[i:(i+gram_len)])
        out.append(new_token)
    return out

def extract_class_labels(label):
    return label.strip('__').lower().split('__')

def sort_key(key):
    return -key[1]

def clean_tokens(tokens, to_replace='[^\w\-\+\&\.\'\"\:]+'):
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    return tokens

def tokenize(mystr):
    mystr = mystr.lower()
    return mystr.split(" ")

def get_tokens(sentence, min_ngram=1, max_ngram=2, to_replace='[^\w\-\+\&\.\'\"\:]+'):
    sentence = re.sub('<[^<]+?>', ' ', sentence)
    sentence = re.sub(to_replace, ' ', sentence)
    
    tokens = clean_tokens(tokenize(sentence), to_replace)
    tokens = [token.strip() for token in tokens]
    
    n_grams = []
    for gram_len in range(min_ngram, max_ngram+1):
        n_grams += ngrams(tokens, gram_len)
    
    return n_grams

def get_features_mi(sentences, labels, num_feats=100, min_ngram=1, max_ngram=2, is_multi_label=True):
    a, b, c = defaultdict(float), defaultdict(float), defaultdict(float)

    total = 0

    for idx in range(len(sentences)):
        sent = sentences[idx]
        label = labels[idx]
        
        common_tokens = set(get_tokens(sent, min_ngram, max_ngram))
        
        total += 1
        
        for token in common_tokens:
            a[token] += 1
            
        if is_multi_label:
            for color in label:
                b[color] += 1
        else:
            b[label] += 1
            
        for token in common_tokens:
            if is_multi_label:
                for color in label:
                    c[(color, token)] += 1
            else:
                c[(label, token)] += 1
    
    mi_values = defaultdict(float)

    for key, val in c.items():
        color, token = key

        x11 = val
        x10 = b[color] - val
        x01 = a[token] - val
        x00 = total - (x11 + x10 + x01)

        x1, x0 = b[color], total - b[color]
        y1, y0 = a[token], total - a[token]

        p = float(x11)/total
        q = float(x10)/total
        r = float(x01)/total
        s = float(x00)/total

        u = float(x1)/total
        v = float(x0)/total
        w = float(y1)/total
        z = float(y0)/total

        a1 = p*np.log2(float(p)/(u*w)) if p != 0 else 0
        a2 = q*np.log2(float(q)/(u*z)) if q != 0 else 0
        a3 = r*np.log2(float(r)/(v*w)) if r != 0 else 0
        a4 = s*np.log2(float(s)/(v*z)) if s != 0 else 0

        mi = a1 + a2 + a3 + a4
        
        mi_values[token] = max(mi_values[token], mi)
    
    final_tokens = [(token, val) for token, val in mi_values.items()]
    final_tokens = sorted(final_tokens, key=sort_key)[:min(num_feats, len(final_tokens))]
    
    final_tokens = [x for x, y in final_tokens]
    
    return final_tokens

def get_filtered_sentence_tokens(sentences, use_features):
    use_features = set(use_features)
    
    tokenized_sentences = [get_tokens(sent) for sent in sentences]
    new_output = []
    for tokens in tokenized_sentences:
        feats = []
        for word in tokens:
            if word in use_features:
                feats.append(word)
        
        new_output.append(feats)
    return new_output

def train_wv_model(sentences, embed_dim, model_path):
    tokenized_sentences = [get_tokens(sent, min_ngram=1, max_ngram=1) for sent in sentences]
    model = Word2Vec(alpha=0.025, size=embed_dim, window=5, min_alpha=0.025, min_count=1, workers=10, negative=10, hs=0, iter=50)

    model.build_vocab(tokenized_sentences)
    model.train(tokenized_sentences, total_examples=model.corpus_count, epochs=50)
    model.save(model_path)
    
    
def train_fasttext_model(sentences, embed_dim, model_path):
    tokenized_sentences = [get_tokens(sent, min_ngram=1, max_ngram=1) for sent in sentences]
    model = FastText(size=embed_dim, window=5, min_count=1, workers=5, iter=50)

    model.build_vocab(tokenized_sentences)
    model.train(tokenized_sentences, total_examples=model.corpus_count, epochs=50)
    model.save(model_path)

def get_weighted_sentence_vectors(sentences, vector_model, idf_scores, vocabulary, vector_dim):
    tokenized_sentences = [get_tokens(sent, min_ngram=1, max_ngram=1) for sent in sentences]
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

def save_data_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
        
def load_data_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)