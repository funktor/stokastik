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
from nltk.corpus import stopwords

def ngrams(tokens, gram_len=2, is_char=False):
    out = []
    for i in range(len(tokens)-gram_len+1):
        new_token = ''.join(tokens[i:(i+gram_len)]) if is_char else ' '.join(tokens[i:(i+gram_len)])
        out.append(new_token)
    return out

def extract_class_labels(label):
    return label.strip('__').lower().split('__')

def sort_key(key):
    return -key[1]

def clean_tokens(tokens, to_replace='[^a-zA-Z0-9\'\.\" ]+'):
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    return tokens

def tokenize(mystr, is_char=False):
#     mystr = mystr.lower()
    return mystr.split(" ") if is_char is False else list(mystr)

def get_tokens(sentence, min_ngram=1, max_ngram=1, to_replace='[^a-zA-Z0-9-\'\.\" ]+', is_char=False):
    sentence = re.sub('<[^<]+?>', ' ', sentence)
    sentence = re.sub(to_replace, ' ', sentence).strip()
    sentence = re.sub('(?<![0-9])\.(?![0-9])|(?<=[0-9])\.(?![0-9])|(?<![0-9])\.(?=[0-9])', ' ', sentence).strip()
    sentence = re.sub('\s+', ' ', sentence)
    
    tokens = clean_tokens(tokenize(sentence, is_char), to_replace)
    tokens = [re.sub('\s+', ' ', token) for token in tokens]
    
    n_grams = []
    for gram_len in range(min_ngram, max_ngram+1):
        n_grams += ngrams(tokens, gram_len, is_char)
    
    if is_char:
        return n_grams
    return [x.strip() for x in n_grams]

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

        a1 = p*np.log2(float(p)/(u*w)) if p > 0 and u != 0 and w != 0 else 0
        a2 = q*np.log2(float(q)/(u*z)) if q > 0 and u != 0 and z != 0 else 0
        a3 = r*np.log2(float(r)/(v*w)) if r > 0 and v != 0 and w != 0 else 0
        a4 = s*np.log2(float(s)/(v*z)) if s > 0 and v != 0 and z != 0 else 0

        mi = a1 + a2 + a3 + a4
        
        mi_values[token] = max(mi_values[token], mi)
    
    final_tokens = [(token, val) for token, val in mi_values.items()]
    final_tokens = sorted(final_tokens, key=sort_key)[:min(num_feats, len(final_tokens))]
    
    final_tokens = [x for x, y in final_tokens]
    
    return final_tokens

def get_features_mi_per_label(sentences, labels, num_feats=100, min_ngram=1, max_ngram=1, is_multi_label=True):
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

        a1 = p*np.log2(float(p)/(u*w)) if p > 0 and u != 0 and w != 0 else 0
        a2 = q*np.log2(float(q)/(u*z)) if q > 0 and u != 0 and z != 0 else 0
        a3 = r*np.log2(float(r)/(v*w)) if r > 0 and v != 0 and w != 0 else 0
        a4 = s*np.log2(float(s)/(v*z)) if s > 0 and v != 0 and z != 0 else 0

        mi = a1 + a2 + a3 + a4
        
        if color not in mi_values:
            mi_values[color]= {}
        
        mi_values[color][token] = mi
    
    final_tokens = {}
    
    for label, values in mi_values.items():
        h = [(k, v) for k, v in values.items()]
        h = sorted(h, key=lambda k:-k[1])[:min(num_feats, len(h))]
        final_tokens[label] = h
    
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
    
def augment_ner_values_with_mi_features(sentences, labels, is_multi_label=False):
    features = get_features_mi_per_label(sentences, labels, max_ngram=3, num_feats=5, is_multi_label=is_multi_label)
    attr_vals = set()

    for tag, feats in features.items():
        tag_tokens = get_tokens(str(tag).lower())
        attr_vals.add(' '.join(tag_tokens).strip())

        p = set(tag_tokens)

        for feat, _ in feats:
            feat_tokens = get_tokens(str(feat).lower())
            q = set(feat_tokens)
            r = min(len(p), len(q))

            if r > 0 and len(p.intersection(q)) == r and len(feat_tokens) > 2:
                attr_vals.add(' '.join(feat_tokens).strip())
    
    return attr_vals

def pre_process_nmt(text, start_token='>', end_token='<'):
    text = [start_token + str(x) + end_token for x in text]
    return text

def custom_classification_scores(true_labels_names, pred_labels_names, valid_labels=None):
    tp, fp, fn = defaultdict(float), defaultdict(float), defaultdict(float)
    support = defaultdict(float)

    for idx in range(len(true_labels_names)):
        true_label, pred_label = list(true_labels_names[idx]), list(pred_labels_names[idx])

        for label in pred_label:
            if valid_labels is None or label in valid_labels:
                if label in true_label:
                    tp[label] += 1
                else:
                    fp[label] += 1

        for label in true_label:
            if valid_labels is None or label in valid_labels:
                support[label] += 1

                if label not in pred_label:
                    fn[label] += 1

    precision, recall, f1_score = defaultdict(float), defaultdict(float), defaultdict(float)

    tot_precision, tot_recall, tot_f1 = 0.0, 0.0, 0.0
    sum_sup = 0.0

    for label, sup in support.items():
        precision[label] = float(tp[label])/(tp[label] + fp[label]) if label in tp and tp[label] + fp[label] != 0 else 0.0
        recall[label] = float(tp[label])/(tp[label] + fn[label]) if label in tp and tp[label] + fn[label] != 0 else 0.0

        f1_score[label] = 2 * float(precision[label] * recall[label])/(precision[label] + recall[label]) if precision[label] + recall[label] != 0 else 0.0

        tot_f1 += sup * f1_score[label]
        tot_precision += sup * precision[label]
        tot_recall += sup * recall[label]

        sum_sup += sup

    for label, sup in support.items():
        print (label, precision[label], recall[label], f1_score[label], sup)

    return (tot_precision/float(sum_sup), tot_recall/float(sum_sup), tot_f1/float(sum_sup), sum_sup) if sum_sup != 0 else (0.0, 0.0, 0.0, 0.0)