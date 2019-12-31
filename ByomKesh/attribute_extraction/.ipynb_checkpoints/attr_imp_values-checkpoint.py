import sys
sys.path.append('/home/jupyter/MySuperMarket')

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
from attribute_extraction.Classifier import Classifier

from collections import defaultdict

def get_features_mi_per_label(sentences, labels, num_feats=100, min_ngram=1, max_ngram=1, is_multi_label=True):
    a, b, c = defaultdict(float), defaultdict(float), defaultdict(float)

    total = 0

    for idx in range(len(sentences)):
        sent = sentences[idx]
        label = labels[idx]
        
        total += 1
        
        a[sent] += 1
            
        if is_multi_label:
            for x in label:
                b[x] += 1
        else:
            b[label] += 1
            
        if is_multi_label:
            for x in label:
                c[(x, sent)] += 1
        else:
            c[(label, sent)] += 1
    
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
        
        if token not in mi_values:
            mi_values[token]= {}
        
        mi_values[token][color] = mi
    
    final_tokens = {}
    
    for label, values in mi_values.items():
        scores = [v for k, v in values.items()]
        pct_value = np.percentile(scores, 80)
        
        h = [(k, v) for k, v in values.items() if v >= pct_value and k in b and b[k] > 2]
        final_tokens[label] = h
    
    return final_tokens

print("Reading data...")
df1 = pd.read_csv('data/ui_mapping.csv')

print("GPT - PT Map...")
gpt_pt_map = {}
pts = list(df1.walmart_pcf_product_type)
gpts = list(df1.walmart_global_product_type)

for i in range(len(pts)):
    gpt_pt_map[gpts[i]] = pts[i]

for pt in set(df1.walmart_pcf_product_type):
    print("Current PT : ", pt)
    
    df_new = df1.loc[df1['walmart_pcf_product_type'] == pt]
    
    msm_attr, wm_attr, msm_attr_type = list(df_new.msm_attribute), list(df_new.walmart_attribute), list(df_new.msm_attribute_type)
    
    f_name = str('_'.join(pt.split()))
    f_name = f_name.replace('/', '_')
    
    if os.path.exists(os.path.join('data', 'product_attributes_new', f_name  + '.tsv')):
        
        print("Filtering WM PTs...")
        df2_new_path = os.path.join('data', 'product_attributes_new', f_name  + '.tsv')
        
        print("Getting attributes...")
        attrs = set(wm_attr)
        
        print("Fetching GPTs...")
        gpts = pputils.get_attribute_vals(df2_new_path, 'global_product_type')
        
        attr_list, gpt_list, values_list, scores_list = [], [], [], []

        print("Fetching important values...")
        for attr in attrs:
            attr_values = pputils.get_attribute_vals(df2_new_path, attr)
            attr_values = [list(set(x.lower().split('__'))) for x in attr_values]

            filtered_vals, filtered_gpts = [], []

            for i in range(len(attr_values)):
                if gpts[i] in gpt_pt_map and gpt_pt_map[gpts[i]] == pt and len(attr_values[i]) > 0 and len(attr_values[i][0]) > 0:
                    filtered_vals.append(attr_values[i])
                    filtered_gpts.append(gpts[i])

            out = get_features_mi_per_label(filtered_gpts, filtered_vals, is_multi_label=True, num_feats=10)

            for g, vs in out.items():
                n = len(vs)
                attr_list += [attr]*n
                gpt_list += [g]*n
                values_list += [x for x, y in vs]
                scores_list += [y for x, y in vs]
                
        df = pd.DataFrame({'GPT':gpt_list, 'ATTR':attr_list, 'VALUE':values_list, 'SCORE':scores_list})
        df.to_csv(os.path.join('outputs', 'attribute_values', f_name + '_attribute_values.csv'))