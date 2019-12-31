import sys
sys.path.append('/home/jupyter/MySuperMarket/')
import pandas as pd
import sklearn_crfsuite, re
import numpy as np
import importlib, os
import logging, math
import json, nltk
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

attributes = ['number_of_lights']

print("Reading data...")
df = pd.read_csv('seating_capacity.csv')

print("Removing None...")
df.dropna(subset=['manual_curation_value'], inplace=True)
df.reset_index(drop=True, inplace=True)

print("Creating data and labels...")
sentences = list(df.title.apply(str).str.lower() + " " + df.short_description.apply(str).str.lower() + " " + df.long_description.apply(str).str.lower())
non_norm_labels = [{attributes[0]:str(x).lower().strip().split('__')} for x in df.manual_curation_value]
non_norm_sentences = [' '.join(x[attributes[0]]) for x in non_norm_labels]
norm_labels = [{attributes[0]:str(x).lower()} for x in list(df.manual_curation_value_normalized.apply(str))]

print("Splitting data into train-test...")
train_indices, valid_indices = train_test_split(range(len(sentences)), test_size=0.2, random_state=0)

train_sentences, valid_sentences = [sentences[i] for i in train_indices], [sentences[i] for i in valid_indices]
train_non_norm_labels, valid_non_norm_labels = [non_norm_labels[i] for i in train_indices], [non_norm_labels[i] for i in valid_indices]
train_non_norm_sentences, valid_non_norm_sentences = [non_norm_sentences[i] for i in train_indices], [non_norm_sentences[i] for i in valid_indices]
train_norm_labels, valid_norm_labels = [norm_labels[i] for i in train_indices], [norm_labels[i] for i in valid_indices]

print("Training 1st level CRF models...")
crf_model_1 = CRFExtractor(attribute_names=attributes, is_multi_label=True)
crf_model_1.train(train_sentences, train_non_norm_labels)

print("Training 2nd level CRF models...")
crf_model_2 = CRFExtractor(attribute_names=attributes, is_multi_label=False)
crf_model_2.train(train_non_norm_sentences, train_norm_labels)

print("Training classifier for 2nd level...")
classifier = Classifier(attribute_names=attributes, is_multi_label=False, num_features=20000, min_ngram=1, max_ngram=3)
classifier.train(train_sentences, train_norm_labels)

print("Predicting with 1st level CRF models...")
preds_1 = crf_model_1.predict(valid_sentences)

print("Predicting with 2nd level CRF models...")
preds_2 = crf_model_2.predict([' '.join(x[attributes[0]]) for x in preds_1])

print("Predicting with classifier...")
preds_classifier = classifier.predict(valid_sentences)

true_labels = [x[attributes[0]] for x in valid_norm_labels]

print("Merging extractor and classifier results...")
pred_labels = []
for x, y in zip(preds_2, preds_classifier):
    if len(x[attributes[0]]) == 0:
        pred_labels.append(y[attributes[0]])
    else:
        if len(set(x[attributes[0]])) > 1:
            pred_labels.append('None')
        else:
            pred_labels.append(x[attributes[0]][0])
            
print(classification_report(true_labels, pred_labels))

print("Using normalizer...")
normalize_module = importlib.import_module('attribute_normalizers.' + attributes[0])
normalizer = normalize_module.Normalizer()

pred_labels2 = normalizer.normalize([x[attributes[0]] for x in preds_1])

print(classification_report(true_labels, pred_labels2))