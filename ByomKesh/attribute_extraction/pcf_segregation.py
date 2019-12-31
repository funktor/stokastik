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

pts = set()
j = 0
for df_chunk in pd.read_csv('data/product_attributes_run_3.tsv', sep='\\t', engine='python', chunksize=50000):
    print(j)
    valid_pcfs = {}
    
    for pt_attr in df_chunk.product_attributes:
        try:
            g = json.loads(pt_attr)

            if 'product_type' in g and 'values' in g['product_type'] and len(g['product_type']['values']) > 0:
                pt = g['product_type']['values'][0]['value']
                if pt not in valid_pcfs:
                    valid_pcfs[pt] = []
                valid_pcfs[pt].append(json.dumps(g))
        except Exception:
            pass
    
    for pt, pt_attrs in valid_pcfs.items():
        df = pd.DataFrame({'product_attributes':pt_attrs})
        f_name = str('_'.join(pt.split())) + '.tsv'
        f_name = f_name.replace('/', '_')
        
        f_name = os.path.join('data', 'product_attributes_og2', f_name)
        with open(f_name, 'a') as f:
            df.to_csv(f, header=f.tell()==0, sep='\t', encoding = 'utf-8')
        
    j += 1