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
                dims = re.findall(r'[0-9\.]+', re.sub('[ -]', '', out_vals[i][j]), re.IGNORECASE)
                
                h = None
                
                if len(dims) == 4:
                    h = dims[0] + '\'' + dims[1] + '"' + ' x ' + dims[2] + '\'' + dims[3] + '"'
                    
                elif len(dims) == 3:
                    p = re.split('x|by', re.sub('[ -]', '', out_vals[i][j]))
                    
                    dims1 = re.findall(r'[0-9\.]+', p[0], re.IGNORECASE)
                    dims2 = re.findall(r'[0-9\.]+', p[1], re.IGNORECASE)
                    
                    if len(dims1) == 2:
                        if 'in' in p[1] or '"' in p[1]:
                            h = dims1[0] + '\'' + dims1[1] + '"' + ' x ' + dims2[0] + '"'
                        elif 'cm' in p[1]:
                            w = float(dims2[0])/2.54
                            h = dims1[0] + '\'' + dims1[1] + '"' + ' x ' + str(round(w, 2)) + '"'
                        else:
                            h = dims1[0] + '\'' + dims1[1] + '"' + ' x ' + dims2[0] + '\''
                            
                    else:
                        if 'in' in p[0] or '"' in p[0]:
                            h = dims1[0] + '"' + ' x ' + dims2[0] + '\'' + dims2[1] + '"'
                        elif 'cm' in p[0]:
                            w = float(dims1[0])/2.54
                            h = str(round(w, 2)) + '"' + ' x ' + dims2[0] + '\'' + dims2[1] + '"'
                        else:
                            h = dims1[0] + '\'' + ' x ' + dims2[0] + '\'' + dims2[1] + '"'
                        
                elif len(dims) == 2:
                    p = re.split('x|by', re.sub('[ -]', '', out_vals[i][j]))
                    
                    if len(p) == 1:
                        dims1 = re.findall(r'[0-9\.]+', p[0], re.IGNORECASE)
                        h = dims1[0] + '\'' + dims1[1] + '"'
                        
                    else:
                        dims1 = re.findall(r'[0-9\.]+', p[0], re.IGNORECASE)
                        dims2 = re.findall(r'[0-9\.]+', p[1], re.IGNORECASE)
                        
                        if 'in' in p[0] or '"' in p[0]:
                            h1 = dims1[0] + '"'
                        elif 'cm' in p[0]:
                            w = float(dims1[0])/2.54
                            h1 = str(round(w, 2)) + '"'
                        else:
                            h1 = dims1[0] + '\''
                            
                        if 'in' in p[1] or '"' in p[1]:
                            h2 = dims2[0] + '"'
                        elif 'cm' in p[1]:
                            w = float(dims2[0])/2.54
                            h2 = str(round(w, 2)) + '"'
                        else:
                            h2 = dims2[0] + '\''
                            
                        h = h1 + ' x ' + h2
                
                elif len(dims) == 1:
                    h = dims[0] + '\''
                
                if h is not None:
#                     g = re.sub(r'[0-9\.x\'\"]', '', out_vals[i][j])
#                     txt = re.findall(r'[a-zA-Z]+', out_vals[i][j], re.IGNORECASE)
#                     if len(txt) > 0:
#                         if len(set(['w', 'l', 'width', 'height', 'length', 'W', 'L']).intersection(set(txt))) > 0:
#                             h += ' '.join(txt)
                            
                    if h not in y1:
                        y1[h] = 0
                    y1[h] += 1
                    
                        
                        
                
#                 y2 = re.findall(r'[0-9\.\']+', re.sub('\'\'', '"', re.sub('[ -]', '', out_vals[i][j])), re.IGNORECASE)
#                 if len(y2) > 0:
#                     if len(y2) > 1:
#                         a, b = y2[0], y2[1]
                        
#                         if 'in' in out_vals[i][j] and a[-1] != '"':
#                             a = a + '"'
                            
#                         if 'in' in out_vals[i][j] and b[-1] != '"':
#                             b = b + '"'
                            
#                         if ('ft' in out_vals[i][j] or 'feet' in out_vals[i][j]) and a[-1] != "'":
#                             a = a + "'"
                            
#                         if ('ft' in out_vals[i][j] or 'feet' in out_vals[i][j]) and b[-1] != "'":
#                             b = b + "'"
                            
#                         if a[-1] != "'" and a[-1] != '"' and 'ft' not in out_vals[i][j] and 'feet' not in out_vals[i][j]:
#                             a = a + '"'
                            
#                         if b[-1] != "'" and b[-1] != '"' and 'ft' not in out_vals[i][j] and 'feet' not in out_vals[i][j]:
#                             b = b + '"'
                            
#                         h = str(a) + "x" + str(b) 
#                     else:
#                         h = out_vals[i][j]
                        
#                     if h not in y1:
#                         y1[h] = 0
#                     y1[h] += 1

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