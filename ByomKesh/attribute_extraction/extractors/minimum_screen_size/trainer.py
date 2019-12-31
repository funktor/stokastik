import pandas as pd
import sklearn_crfsuite, re
import numpy as np
import importlib, os
import logging, math
import json, nltk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import Utilities as utils
from attribute_extraction.BIOEncoding import BIOEncoder
from attribute_extraction.CRF_Extractor import CRFExtractor
from attribute_extraction.Classifier import Classifier
from attribute_extraction.extractors.minimum_screen_size.normalizer import Normalizer
from attribute_extraction.Neural_Machine_Translation import NMTNetwork
from attribute_extraction.NMT_Driver import NMTDriver

attributes = ['minimum_screen_size']
use_seq_to_seq = True

print("Reading data...")
df = pd.read_csv('data/minimum_screen_size.csv')

# print("Removing None...")
# df.dropna(subset=['manual_curation_value'], inplace=True)
# df.reset_index(drop=True, inplace=True)

print(df.shape)

print("Creating data and labels...")
sentences = list(df.title.apply(str).str.lower() + " " + df.short_description.apply(str).str.lower() + " " + df.long_description.apply(str).str.lower())
sentences = [' '.join(utils.get_tokens(x)) for x in sentences]

non_norm_labels = [{attributes[0]:str(x).lower().strip().split('__')} for x in df.manual_curation_value]
non_norm_sentences = [' '.join(utils.get_tokens(x[attributes[0]][0])) for x in non_norm_labels]
norm_labels = [{attributes[0]:str(x).lower()} for x in list(df.manual_curation_value_normalized.apply(str))]

print("Splitting data into train-test...")
train_indices, valid_indices = train_test_split(range(len(sentences)), test_size=0.2, random_state=0)

train_sentences, valid_sentences = [sentences[i] for i in train_indices], [sentences[i] for i in valid_indices]
train_non_norm_labels, valid_non_norm_labels = [non_norm_labels[i] for i in train_indices], [non_norm_labels[i] for i in valid_indices]
train_non_norm_sentences, valid_non_norm_sentences = [non_norm_sentences[i] for i in train_indices], [non_norm_sentences[i] for i in valid_indices]
train_norm_labels, valid_norm_labels = [norm_labels[i] for i in train_indices], [norm_labels[i] for i in valid_indices]

print("Training 1st level CRF models...")
seq_model_1 = CRFExtractor(attribute_names=attributes, is_multi_label=True)
seq_model_1.train(train_sentences, train_non_norm_labels)

print("Saving 1st level CRF models...")
utils.save_data_pkl(seq_model_1, 'models/seq_model_1.pkl')

if use_seq_to_seq:
    print("Training seq2seq model for 2nd level inference...")
    seq_model_2 = NMTDriver(use_char=True)
    seq_model_2.train(train_non_norm_sentences, train_norm_labels)
else:
    print("Training 2nd level CRF models...")
    seq_model_2 = CRFExtractor(attribute_names=attributes, is_multi_label=False)
    seq_model_2.train(train_non_norm_sentences, train_norm_labels)

print("Saving 2nd level CRF models...")
utils.save_data_pkl(seq_model_2, 'models/seq_model_2.pkl')

print("Training classifier for 2nd level...")
classifier = Classifier(attribute_names=attributes, is_multi_label=False, num_features=20000, min_ngram=1, max_ngram=3)
classifier.train(train_sentences, train_norm_labels)

print("Saving classifier...")
utils.save_data_pkl(classifier, 'models/classifier.pkl')

print("Predicting with 1st level CRF models...")
preds_1 = seq_model_1.predict(valid_sentences)

print("Predicting with 2nd level models...")
preds_2 = seq_model_2.predict([' '.join(utils.get_tokens(x[attributes[0]][0])) if len(x[attributes[0]]) > 0 else '' for x in preds_1])
preds_2 = [preds_2[i] if len(preds_1[i][attributes[0]]) > 0 else {attributes[0]:''} for i in range(len(preds_2))]

print("Predicting with classifier...")
preds_classifier = classifier.predict(valid_sentences)

true_labels = [x[attributes[0]] for x in valid_norm_labels]

print("Merging extractor and classifier results...")
pred_labels = []
for x, y in zip(preds_2, preds_classifier):
    if len(x[attributes[0]]) == 0:
        pred_labels.append(y[attributes[0]])
    else:
        pred_labels.append(x[attributes[0]])
            
print(classification_report(true_labels, pred_labels))

print("Using normalizer...")
normalizer = Normalizer()
pred_labels2 = normalizer.normalize([x[attributes[0]] for x in preds_1])
print(classification_report(true_labels, pred_labels2))