import pandas as pd
import numpy as np
import os, re, math, json, nltk, copy
import Utilities as utils
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from SuffixTree import SuffixTree
from attribute_extraction.Normalizer import BaseNormalizer

class Normalizer(BaseNormalizer):
    def __init__(self):
        super(Normalizer, self).__init__()
        
    def normalize(self, out_vals):
        pred_labels = []
        for i in range(len(out_vals)):
            y1 = {}
            for j in range(len(out_vals[i])):
                y2 = re.findall(r'\bone|two|three|four|five|six|seven|eight|nine|ten|single|eleven|twelve|single|twin|triple\b', out_vals[i][j], re.IGNORECASE)

                if len(y2) > 0:
                    for k in y2:
                        h = None
                        if k.lower() == 'one' or k.lower() == 'single':
                            h = 1
                        elif k.lower() == 'two' or k.lower() == 'twin':
                            h = 2
                        elif k.lower() == 'three' or k.lower() == 'triple':
                            h = 3
                        elif k.lower() == 'four':
                            h = 4
                        elif k.lower() == 'five':
                            h = 5
                        elif k.lower() == 'six':
                            h = 6
                        elif k.lower() == 'seven':
                            h = 7
                        elif k.lower() == 'eight':
                            h = 8
                        elif k.lower() == 'nine':
                            h = 9
                        elif k.lower() == 'ten':
                            h = 10
                        elif k.lower() == 'eleven':
                            h = 11
                        elif k.lower() == 'twelve':
                            h = 12

                        if h is not None:
                            if h not in y1:
                                y1[h] = 0
                            y1[h] += 1

            if len(y1) == 0:
                for j in range(len(out_vals[i])):
                    y2 = re.findall(r'(\d+)[^a-zA-Z]*lights?|lights?[^a-zA-Z]*?(\d+)| (\d+)[^a-zA-Z]*bulbs?|bulbs?[^a-zA-Z]*?(\d+)|(\d+)\s*x|\((\d+)\)|(\d+)\-|(\d+)\s+\d+\s*W|(\d+)\s*LED|(\d+)\s*Pack|(\d+)\s*Lt|(\d+)\s*pcs?', 
                                    out_vals[i][j], re.IGNORECASE)
                    for h1 in y2:
                        for h in h1:
                            if h != '' and int(h) != 0:
                                h = int(h)
                                if h not in y1:
                                    y1[h] = 0
                                y1[h] += 1

                    y2 = re.findall(r'\b\d+\b', out_vals[i][j], re.IGNORECASE)
                    for h in y2:
                        if h != '' and int(h) != 0:
                            h = int(h)
                            if h not in y1:
                                y1[h] = 0
                            y1[h] += 1

            if len(y1) == 0:
                pred_labels.append('None')

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
                    pred_labels.append('None')

        return pred_labels