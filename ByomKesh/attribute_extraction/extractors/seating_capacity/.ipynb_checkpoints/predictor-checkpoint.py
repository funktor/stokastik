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

WALMART_ORG_ID = "f55cdc31-ab75-4bb6-8fe0-b39041159d63"

class Predictor():
    def __init__(self, use_normalizer=False):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        
        self.classifier = utils.load_data_pkl(os.path.join(curr_path, 'models', 'classifier.pkl'))
        self.use_normalizer = use_normalizer
        self.crf_model_1 = utils.load_data_pkl(os.path.join(curr_path, 'models', 'crf_model_1.pkl'))
        self.crf_model_2 = utils.load_data_pkl(os.path.join(curr_path, 'models', 'crf_model_2.pkl'))
        
        
    def predict_all(self, inputs):
        attributes = ['seating_capacity']
        
        logging.info("Received Request: %s", inputs)
        
        try:
            host_ip = socket.gethostbyname(socket.gethostname())
        except:
            host_ip = 'unknown'
            
        inputs = [json.loads(input) for input in inputs]
        sentences = [input['title'] + " " + input['short_description'] + " " + input['long_description'] for input in inputs]
        
        print("Predicting with 1st level CRF models...")
        preds_1 = self.crf_model_1.predict(sentences)
        
        if self.use_normalizer is False:
            print("Predicting with 2nd level CRF models...")
            preds_2 = self.crf_model_2.predict([' '.join(x[attributes[0]]) for x in preds_1])

            print("Predicting with classifier...")
            preds_classifier = self.classifier.predict(sentences)

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
        
        else:
            print("Using normalizer...")
            normalize_module = importlib.import_module('attribute_extraction.attribute_normalizers.' + attributes[0])
            normalizer = normalize_module.Normalizer()

            pred_labels = normalizer.normalize([x[attributes[0]] for x in preds_1])
        
        print("Creating response...")
        
        response = [json.dumps({"host":host_ip, 
                    "predictions":[{"org_id":WALMART_ORG_ID, 
                                    "status":"OK", 
                                    "attribute":attributes[0], 
                                    "prediction":[pred_labels[i]], 
                                    "probability":[1.0]
                                   }]}) for i in range(len(pred_labels))]
            
        return response

