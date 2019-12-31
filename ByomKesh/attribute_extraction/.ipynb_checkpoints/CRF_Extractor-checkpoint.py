import pandas as pd
import numpy as np
import os, re, math, json, nltk
import Utilities as utils
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from attribute_extraction.SuffixTree import SuffixTree
from attribute_extraction.BIOEncoding import BIOEncoder

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word_position_window': int(i/5)+1,
        'word_position_rel_window': i%5,
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features['BOS_WORD'] = False
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS_WORD'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features['EOS_WORD'] = False
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS_WORD'] = True

    return features


def char2features(sent, i):
    char = sent[i][0]

    features = {
        'bias': 1.0,
        'char.lower()': char.lower(),
        'char.isupper()': char.isupper(),
        'char.istitle()': char.istitle(),
        'char.isdigit()': char.isdigit(),
        'char_position_window': int(i/5)+1,
        'char_position_rel_window': i%5,
    }
    if i > 0:
        char1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features['BOS_CHAR'] = False
        features.update({
            '-1:char.lower()': char1.lower(),
            '-1:char.istitle()': char1.istitle(),
            '-1:char.isupper()': char1.isupper(),
        })
    else:
        features['BOS_CHAR'] = True

    if i < len(sent)-1:
        char1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features['EOS_CHAR'] = False
        features.update({
            '+1:char.lower()': char1.lower(),
            '+1:char.istitle()': char1.istitle(),
            '+1:char.isupper()': char1.isupper(),
        })
    else:
        features['EOS_CHAR'] = True

    return features


def sent2word_features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2char_features(sent):
    return [char2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

class CRFExtractor():
    def __init__(self, attribute_names, max_word_length=150, max_char_length=500, attr_values_map=None, is_multi_label=False, use_char_crf=True):
        self.attribute_names = attribute_names
        self.is_multi_label = is_multi_label
        self.attr_values_map = attr_values_map
        self.max_word_length = max_word_length
        self.max_char_length = max_char_length
        self.use_char_crf = use_char_crf
        
        self.bioe_encoder = BIOEncoder(max_word_length=max_word_length, 
                                       max_char_length=max_char_length, attr_values_map=attr_values_map, is_multi_label=is_multi_label)
    
    def train(self, sentences, sent_pcf_labels=None):
        train_sents_words, train_sents_chars = self.bioe_encoder.bioe_encoding(sentences, sent_pcf_labels=sent_pcf_labels, mode='Train')

        X_train_words = [sent2word_features(s) for s in train_sents_words]
        y_train_words = [sent2labels(s) for s in train_sents_words]
        
        print("Training word level CRF...")
        self.word_crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
        self.word_crf.fit(X_train_words, y_train_words)
        
        if self.use_char_crf:
            X_train_chars = [sent2char_features(s) for s in train_sents_chars]
            y_train_chars = [sent2labels(s) for s in train_sents_chars]

            print("Training char level CRF...")
            self.char_crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
            self.char_crf.fit(X_train_chars, y_train_chars)
        
        
    def predict(self, sentences):
        valid_sents_words, valid_sents_chars = self.bioe_encoder.bioe_encoding(sentences, sent_pcf_labels=None, mode='Test')

        X_valid_words = [sent2word_features(s) for s in valid_sents_words]
        y_valid_words = [sent2labels(s) for s in valid_sents_words]
        predictions_words = self.word_crf.predict(X_valid_words)
        
        if self.use_char_crf:
            X_valid_chars = [sent2char_features(s) for s in valid_sents_chars]
            y_valid_chars = [sent2labels(s) for s in valid_sents_chars]
            predictions_chars = self.char_crf.predict(X_valid_chars)
        
        results = [{} for i in range(len(sentences))]

        for attr in self.attribute_names:
            out_vals_words = []
            b_tag, i_tag, s_tag, e_tag = 'B-' + attr, 'I-' + attr, 'S-' + attr, 'E-' + attr

            for i in range(len(predictions_words)):
                curr_val, all_curr_val = [], []

                for j in range(len(predictions_words[i])):
                    q = predictions_words[i][j]
                    
                    if len(q) > 2:
                        if q == b_tag or q == s_tag:
                            if len(curr_val) > 0:
                                y = ' '.join(curr_val)
                                all_curr_val.append(y)
                            curr_val = [valid_sents_words[i][j][0]]

                        elif q == i_tag or q == e_tag:
                            curr_val += [valid_sents_words[i][j][0]]

                if len(curr_val) > 0:
                    y = ' '.join(curr_val)
                    all_curr_val.append(y)

                if len(all_curr_val) > 0:
                    out_vals_words.append(list(set(all_curr_val)))
                else:
                    out_vals_words.append([])
            
            if self.use_char_crf:
                out_vals_chars = []
                
                for i in range(len(predictions_chars)):
                    curr_val, all_curr_val = [], []

                    for j in range(len(predictions_chars[i])):
                        q = predictions_chars[i][j]

                        if len(q) > 2:
                            if q == b_tag or q == s_tag:
                                if len(curr_val) > 0:
                                    y = ''.join(curr_val)
                                    all_curr_val.append(y)
                                curr_val = [valid_sents_chars[i][j][0]]

                            elif q == i_tag or q == e_tag:
                                curr_val += [valid_sents_chars[i][j][0]]

                    if len(curr_val) > 0:
                        y = ''.join(curr_val)
                        all_curr_val.append(y)

                    if len(all_curr_val) > 0:
                        out_vals_chars.append(list(set(all_curr_val)))
                    else:
                        out_vals_chars.append([])

            for i in range(len(out_vals_words)):
                if self.use_char_crf and len(out_vals_words[i]) == 0:
                    out_vals_words[i] = out_vals_chars[i]
                    
                if self.is_multi_label:
                    for x in attr.split('__'):
                        if x not in results[i]:
                            results[i][x] = []
                        results[i][x] += out_vals_words[i]
                else:
                    for x in attr.split('__'):
                        results[i][x] = out_vals_words[i][0] if len(out_vals_words[i]) > 0 else ''
            
        return results