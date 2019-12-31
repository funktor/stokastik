import pandas as pd
import numpy as np
import os, re, math, json, nltk
import Utilities as utils
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import attribute_extraction.PreProcessingUtils as pputils
from attribute_extraction.CRF_Extractor import CRFExtractor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import itertools

class Classifier():
    def __init__(self, attribute_names, is_multi_label=False, num_features=1000, min_ngram=1, max_ngram=1):
        self.attribute_names = attribute_names
        self.vectorizer = {}
        self.label_transformer = {}
        self.model = {}
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.is_multi_label = is_multi_label
        self.num_features = num_features
    
    def train(self, sentences, sent_pcf_labels=None):
        indices = {}
        for i in range(len(sentences)):
            attr_vals = sent_pcf_labels[i]
            for attr, values in attr_vals.items():
                if len(values) > 0:
                    if attr not in indices:
                        indices[attr] = []
                    indices[attr].append(i)
        
        for attr in self.attribute_names:
            if attr in indices:
                c_indices = indices[attr]
                sents, labels = [sentences[x] for x in c_indices], [sent_pcf_labels[x][attr] for x in c_indices]
                unique_labels = set(itertools.chain(*labels)) if self.is_multi_label else set(labels)
                    
                if len(unique_labels) > 1:
                    corpus_features = utils.get_features_mi(sents, labels, self.num_features, self.min_ngram, self.max_ngram, 
                                                            is_multi_label=self.is_multi_label)

                    self.vectorizer[attr] = TfidfVectorizer(tokenizer=utils.get_tokens, 
                                                            ngram_range=(self.min_ngram, self.max_ngram), 
                                                            stop_words='english', vocabulary=corpus_features)

                    X_data = self.vectorizer[attr].fit_transform(sents)

                    if self.is_multi_label:
                        self.label_transformer[attr] = MultiLabelBinarizer()
                    else:
                        self.label_transformer[attr] = LabelEncoder()

                    y_data = self.label_transformer[attr].fit_transform(labels)

                    if self.is_multi_label:
                        base_model = LogisticRegression(penalty='l1', C=10.0)
                        self.model[attr] = OneVsRestClassifier(base_model)
                    else:
                        self.model[attr] = LogisticRegression(penalty='l1', C=10.0)

                    self.model[attr].fit(X_data, y_data)
                    
                else:
                    self.model[attr] = None
                
            else:
                self.model[attr] = None
        
    def predict(self, sentences):
        predictions = [{} for i in range(len(sentences))]
        
        for attr in self.attribute_names:
            if attr in self.model and self.model[attr] is not None:
                X_data = self.vectorizer[attr].transform(sentences)
                preds = self.model[attr].predict(X_data)
                preds = self.label_transformer[attr].inverse_transform(preds)

                for i in range(len(preds)):
                    for x in attr.split('__'):
                        if self.is_multi_label:
                            if x not in predictions[i]:
                                predictions[i][x] = []
                            predictions[i][x] += list(preds[i])
                        else:
                            predictions[i][x] = preds[i]
        
        return predictions