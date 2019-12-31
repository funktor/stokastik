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

msm_file, wm_file = sys.argv[1], sys.argv[2]
msm_wm_map_file = sys.argv[3]

print("Reading data...")

df1 = pd.read_csv(msm_file)
df2 = pd.read_csv(wm_file, sep='\\t', engine='python')
df_map = pd.read_csv(msm_wm_map_file)

msm_pf, wm_pt = list(df_map.MSM_PF), list(df_map.WM_PT)

for pf in set(msm_pf):
    print("Current PF : ", pf)
    
    df_new = df_map.loc[df_map['MSM_PF'] == pf]
    df1_new = df1.loc[df1['PFName'] == pf]
    
    msm_attr, wm_attr, msm_attr_type = list(df_new.MSM_ATTR), list(df_new.WM_ATTR), list(df_new.MSM_ATTR_TYPE)
    
    print("Filtering WM PTs...")
    df2_new = pputils.filter_wm_pcf_pt(df2, df_new.WM_PT)
    
    print("Creating MSM to WM attribute map...")
    msm_wm_map, wm_msm_map = pputils.generate_msm_wm_attr_map(msm_attr, wm_attr)
    
    print("Getting classifier flags...")
    u_classifier = pputils.use_classifier(df2_new, wm_attr, wm_msm_map)
    
    print("Creating MSM attribute type map...")
    msm_type_map = pputils.generate_msm_type_map(msm_attr, msm_attr_type)
    
    print("Creating MSM attributes to values map...")
    msm_attr_vals = pputils.generate_msm_attr_values(df1_new, msm_attr, msm_type_map)
    
    print("Getting WM PCF title, description and labels...")
    wm_sentences, wm_pcf_labels = pputils.generate_wm_data_labels(df2_new, wm_attr, wm_msm_map)
    
    print("Getting WM attribute value counts...")
    wm_attr_value_counts = pputils.generate_wm_attr_value_counts(df2_new, wm_attr)
    
    print("Creating WM attributes to values map...")
    wm_attr_vals = pputils.generate_wm_attr_values_normalized(wm_sentences, wm_pcf_labels, set(msm_attr), wm_attr_value_counts, min_count=10)
    
    print("Creating train-test indices...")
    train_indices, valid_indices = train_test_split(range(len(wm_sentences)), test_size=0.2, random_state=0)
    
    print("Merging MSM and WM attribute value maps...")
    combined_attr_vals = pputils.combine_msm_attr_values(msm_wm_map, msm_attr_vals, wm_attr_vals)
    
    print("Initializing extractor...")
    extractor = CRFExtractor(attribute_names=list(set(msm_attr)), attr_values_map=combined_attr_vals, is_multi_label=True)
    
    print("Initializing classifier...")
    classifier = Classifier(attribute_names=list(set(msm_attr)), is_multi_label=True, num_features=20000, min_ngram=1, max_ngram=3)
    
    print("Creating training/validation data...")
    train_sentences, valid_sentences = [wm_sentences[i] for i in train_indices], [wm_sentences[i] for i in valid_indices]
    train_labels, valid_labels = [wm_pcf_labels[i] for i in train_indices], [wm_pcf_labels[i] for i in valid_indices]
        
    print(len(train_sentences), len(valid_sentences))
    
    print("Training extractor...")
    extractor.train(train_sentences, train_labels)
    
    print("Training classifier...")
    classifier.train(train_sentences, train_labels)
    
    print("Fetching validation results...")
    extraction_results = extractor.predict(valid_sentences)
    classification_results = classifier.predict(valid_sentences)
    
    print("Merging extraction and classifier results...")
    preds = pputils.merge_extractor_and_classifier_results(extraction_results, classification_results, list(set(msm_attr)), u_classifier)
    
    print("Defaulting to PCF values in case of no result...")
    preds = pputils.fillback_with_pcf(preds, valid_labels, list(set(msm_attr)))
    
    print("Saving extractor...")
    utils.save_data_pkl(extractor, os.path.join('models_metadata', '_'.join(utils.get_tokens(pf)) + '_extractor.pkl'))
    
    print("Saving classifier...")
    utils.save_data_pkl(classifier, os.path.join('models_metadata', '_'.join(utils.get_tokens(pf)) + '_classifier.pkl'))
    
    print("Writing output...")
    results = {}
    for i in range(len(preds)):
        for attr, vals in preds[i].items():
            if attr not in results:
                results[attr] = []
            results[attr].append(vals)
            
    results['wm_desc'] = valid_sentences
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join('outputs', '_'.join(utils.get_tokens(pf)) + '_extracted.csv'))