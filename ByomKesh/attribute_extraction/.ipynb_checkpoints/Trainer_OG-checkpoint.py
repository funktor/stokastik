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

use_pre_trained = True

print("Reading data...")

df1 = pd.read_csv('data/ui_mapping_new.csv')

with open('data/iron_bank_flavors.txt', encoding='utf-8') as f:
    ib_flavors = f.readlines()

ib_flavors = set([x.strip().lower() for x in ib_flavors])

for pt in set(df1.walmart_pcf_product_type):
    print("Current PT : ", pt)
    
    df_new = df1.loc[df1['walmart_pcf_product_type'] == pt]
    
    msm_attr, wm_attr, msm_attr_type = list(df_new.msm_attribute), list(df_new.walmart_attribute), list(df_new.msm_attribute_type)
    
    f_name = str('_'.join(pt.split()))
    f_name = f_name.replace('/', '_')
    
    if os.path.exists(os.path.join('data', 'product_attributes_new', f_name  + '.tsv')) and os.path.join('data', 'product_attributes_og2', f_name + '.tsv'):
        print("Filtering WM PTs...")
        df2_new_path = os.path.join('data', 'product_attributes_new', f_name  + '.tsv')

        print("Creating MSM to WM attribute map...")
        msm_wm_map, wm_msm_map = pputils.generate_msm_wm_attr_map(msm_attr, wm_attr)

        print("Getting classifier flags...")
        u_classifier = pputils.use_classifier(df2_new_path, wm_attr, wm_msm_map)

        print("Creating MSM attribute type map...")
        msm_type_map = pputils.generate_msm_type_map(msm_attr, msm_attr_type)

        print("Creating MSM attributes to values map...")
        msm_attr_vals = pputils.generate_msm_attr_values_closed_list(df_new, msm_attr, msm_type_map)

        print("Getting WM PCF title, description and labels...")
        wm_sentences, wm_pcf_labels = pputils.generate_wm_data_labels(df2_new_path, wm_attr, wm_msm_map)

        print("Getting WM attribute value counts...")
        wm_attr_value_counts = pputils.generate_wm_attr_value_counts(df2_new_path, wm_attr)

        print("Creating WM attributes to values map...")
        wm_attr_vals = pputils.generate_wm_attr_values_closed_list(df_new, wm_attr, wm_msm_map)
#         wm_attr_vals = pputils.generate_wm_attr_values_normalized(wm_sentences, wm_pcf_labels, 
#                                                                   list(u_classifier.keys()), wm_attr_value_counts, min_count=5)
        
#         wm_attr_vals_mapping = pputils.generate_wm_attr_values_closed_list(df_new, wm_attr, wm_msm_map)
#         wm_attr_vals = pputils.merge_walmart_values(wm_attr_vals_mapping, wm_attr_vals)

        if use_pre_trained and os.path.exists(os.path.join('models_metadata', f_name + '_extractor.pkl')) and os.path.exists(os.path.join('models_metadata', f_name + '_classifier.pkl')):
            print("Loading extractor...")
            extractor = utils.load_data_pkl(os.path.join('models_metadata', f_name + '_extractor.pkl'))
            
            print("Loading classifier...")
            classifier = utils.load_data_pkl(os.path.join('models_metadata', f_name + '_classifier.pkl'))
            
        else:
            print("Merging MSM and WM attribute value maps...")
            combined_attr_vals = pputils.combine_msm_attr_values(msm_wm_map, msm_attr_vals, wm_attr_vals)
            
            if 'TOV Flavour' in combined_attr_vals:
                combined_attr_vals['TOV Flavour'] += list(ib_flavors)

            print("Initializing extractor...")
            extractor = CRFExtractor(attribute_names=list(u_classifier.keys()), attr_values_map=combined_attr_vals, 
                                     is_multi_label=True, use_char_crf=False)

            print("Initializing classifier...")
            classifier = Classifier(attribute_names=list(u_classifier.keys()), is_multi_label=True, num_features=20000, min_ngram=1, max_ngram=3)

            print("Training extractor...")
            extractor.train(wm_sentences, wm_pcf_labels)

            print("Training classifier...")
            classifier.train(wm_sentences, wm_pcf_labels)
            
            print("Saving extractor...")
            utils.save_data_pkl(extractor, os.path.join('models_metadata', f_name + '_extractor.pkl'))

            print("Saving classifier...")
            utils.save_data_pkl(classifier, os.path.join('models_metadata', f_name + '_classifier.pkl'))

        print("Getting OG dataframe...")
        df2_new_pred_path = os.path.join('data', 'product_attributes_og2', f_name + '.tsv')

        print("Getting OG title, description and labels...")
        wm_og_sentences, wm_og_pcf_labels = pputils.generate_wm_data_labels(df2_new_pred_path, wm_attr, wm_msm_map)

        print("Fetching OG prediction results...")
        extraction_results = extractor.predict(wm_og_sentences)
        classification_results = classifier.predict(wm_og_sentences)

        print("Merging extraction and classifier results...")
        preds = pputils.merge_extractor_and_classifier_results(extraction_results, classification_results, list(set(msm_attr)), u_classifier)

#         print("Defaulting to PCF values in case of no result...")
#         preds = pputils.fillback_with_pcf(preds, wm_og_pcf_labels, list(set(msm_attr)))

        print("Processing boolean attributes...")
        preds = pputils.process_boolean_attributes(preds, bool_attrs=set(['Organic']))

        print("Writing output...")
        results = {}
        for i in range(len(preds)):
            for attr, vals in preds[i].items():
                if attr not in results:
                    results[attr] = []
                results[attr].append(vals)

        results['wm_desc'] = wm_og_sentences
        results['global_product_type'] = pputils.get_attribute_vals(df2_new_pred_path, 'global_product_type')
        
        df_out = pd.DataFrame(results)
        df_out.to_csv(os.path.join('outputs', f_name + '_extracted_no_pcf.csv'))

#         print("Getting attribute values...")
#         pid_list = pputils.get_attribute_vals(df2_new_pred_path, 'item_id')
#         pt_list = pputils.get_attribute_vals(df2_new_pred_path, 'product_type')
#         gpt_list = pputils.get_attribute_vals(df2_new_pred_path, 'global_product_type')

#         oga_list = [json.dumps(x) for x in preds]

#         out_df = pd.DataFrame({'product_id':pid_list, 'product_type':pt_list, 'global_product_type':gpt_list, 'og_attributes':oga_list})

#         out_df.to_csv(os.path.join('outputs', 'product_attributes_og', f_name + '.csv'), sep=',')