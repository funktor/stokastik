import pandas as pd
import numpy as np
import os, re, math, json, nltk, copy
import Utilities as utils
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from attribute_extraction.SuffixTree import SuffixTree
from attribute_extraction.Normalizer import BaseNormalizer

class Normalizer(BaseNormalizer):
    def __init__(self):
        super(Normalizer, self).__init__()
        
    def normalize(self, out_vals):
        pred_labels = []
        for i in range(len(out_vals)):
            y1 = {}

            for j in range(len(out_vals[i])):
                y2 = re.findall(r"[-+]?\d*\.\d+|\d+", out_vals[i][j], re.IGNORECASE)
                if len(y2) > 0:
                    h = y2[0]
                    if h not in y1:
                        y1[h] = 0
                    y1[h] += 1

            if len(y1) == 0:
                pred_labels.append('none')

            elif len(y1) == 1:
                pred_labels.append(str(list(y1.keys())[0]))

            else:
                max_cnt, max_v = 1, None
                for v, cnt in y1.items():
                    if cnt > max_cnt:
                        max_cnt = cnt
                        max_v = v

                if max_v is not None:
                    pred_labels.append(str(max_v))
                else:
                    pred_labels.append('none')
                    
        return pred_labels 