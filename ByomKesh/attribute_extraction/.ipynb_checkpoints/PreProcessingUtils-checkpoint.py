import pandas as pd
import numpy as np
import os, re, math, json, nltk
import Utilities as utils
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import mysql.connector
import itertools

def generate_msm_wm_attr_map(msm_attr, wm_attr):
    msm_wm_map, wm_msm_map = {}, {}

    for i in range(len(msm_attr)):
        if msm_attr[i] not in msm_wm_map:
            msm_wm_map[msm_attr[i]] = set()
        msm_wm_map[msm_attr[i]].add(wm_attr[i])
        
        if wm_attr[i] not in wm_msm_map:
            wm_msm_map[wm_attr[i]] = set()
        wm_msm_map[wm_attr[i]].add(msm_attr[i])
        
    for wm_attr, m_attrs in wm_msm_map.items():
        wm_msm_map[wm_attr] = '__'.join(list(m_attrs))
    
    return msm_wm_map, wm_msm_map

def generate_msm_type_map(msm_attr, msm_attr_type):
    return {msm_attr[i]:msm_attr_type[i] for i in range(len(msm_attr))}

def use_classifier(wm_df_path, wm_attr, wm_msm_map):
    u_classifier = {}
    unique_vals, vals_in_text, attr_cnts = {}, {}, {}
    
    for wm_df in pd.read_csv(wm_df_path, sep='\t', encoding='utf-8', chunksize=5000):
        for pd_attr in wm_df.product_attributes:
            g = json.loads(pd_attr)

            title = g['product_name']['values'][0]['value'] if 'product_name' in g else ''
            short_desc = g['product_short_description']['values'][0]['value'] if 'product_short_description' in g else ''
            long_desc = g['product_long_description']['values'][0]['value'] if 'product_long_description' in g else ''

            corpus = title + " " + short_desc + " " + long_desc
            corpus = ' '.join(utils.get_tokens(corpus.lower()))
            corpus = re.sub('\s+', ' ', corpus)

            for attr in set(wm_attr):
                if attr in g:
                    v = g[attr]
                    if attr in wm_msm_map:
                        attr = wm_msm_map[attr]

                        if attr not in attr_cnts:
                            attr_cnts[attr] = 0
                        attr_cnts[attr] += 1

                        if 'values' in v and len(v['values']) > 0:
                            for j in range(len(v['values'])):
                                if 'value' in v['values'][j]:
                                    tokens = utils.get_tokens(str(v['values'][j]['value']).lower())
                                    x = ' '.join(tokens).strip()
                                    x = re.sub('\s+', ' ', x)
                                    if len(x) > 0:
                                        if attr not in unique_vals:
                                            unique_vals[attr] = set()
                                        unique_vals[attr].add(x)

                                        if x in corpus:
                                            if attr not in vals_in_text:
                                                vals_in_text[attr] = 0
                                            vals_in_text[attr] += 1
    
    for attr, cnts in attr_cnts.items():
        a = attr not in unique_vals or cnts/len(unique_vals[attr]) < 10
        b = attr not in vals_in_text or vals_in_text[attr]/cnts > 0.60
        
        if a or b:
            u_classifier[attr] = False
        else:
            u_classifier[attr] = True
    
    return u_classifier

def generate_msm_attr_values(msm_df, msm_attr, msm_type_map):
    msm_attr_vals = {}
    
    for attr in set(msm_attr):
        df = msm_df.loc[msm_df['AttrName'] == attr]
        if df.shape[0] > 0:
            if msm_type_map[attr] == 'Boolean' or len(set(['yes', 'no', 'y', 'n']).intersection(set(df.AttrVal.str.lower()))) > 0:
                msm_attr_vals[attr] = set([' '.join(utils.get_tokens(str(attr).lower())).strip()])
            else:
                tags = list(df.AttrVal.str.lower())
                corpus = list(df.PDName.apply(str).str.lower() + " " + df.PDDesc.apply(str).str.lower())
                
                msm_attr_vals[attr] = utils.augment_ner_values_with_mi_features(corpus, tags, is_multi_label=False)
    
    return msm_attr_vals

def generate_msm_attr_values_closed_list(wm_msm_map_df, msm_attr, msm_type_map):
    msm_attr_vals = {}
    
    for attr in set(msm_attr):
        df = wm_msm_map_df.loc[wm_msm_map_df['msm_attribute'] == attr]
        if df.shape[0] > 0:
            attr_values = list(df.msm_values.apply(str).str.lower())
            attr_values = [x.split(';') for x in attr_values]
            attr_values = set(itertools.chain(*attr_values))
            
            if msm_type_map[attr] == 'BOOLEAN' or len(set(['yes', 'no', 'y', 'n']).intersection(attr_values)) > 0:
                msm_attr_vals[attr] = set([' '.join(utils.get_tokens(str(attr).lower())).strip()])
                                                      
            else:
                msm_attr_vals[attr] = set([' '.join(utils.get_tokens(x)) for x in attr_values if len(x) > 0])
    
    return msm_attr_vals

def filter_wm_pcf_pt(wm_df_path, wm_pts):
    indices = []
    
    for wm_df in pd.read_csv(wm_df_path, sep='\t', encoding='utf-8', chunksize=5000):
        for i in range(len(wm_df.product_attributes)):
            g = json.loads(wm_df.product_attributes[i])

            if 'product_type' in g and 'values' in g['product_type'] and len(g['product_type']['values']) > 0:
                if g['product_type']['values'][0]['value'] in set(wm_pts):
                    indices.append(i)
    
    return wm_df.iloc[indices,:].reset_index(drop=True)

def transform_pcf(wm_df_path):
    title, short_desc, long_desc = [], [], []
    
    for wm_df in pd.read_csv(wm_df_path, sep='\t', encoding='utf-8', chunksize=5000):
        for pd_attr in wm_df.product_attributes:
            g = json.loads(pd_attr)

            title += [g['product_name']['values'][0]['value'] if 'product_name' in g else '']
            short_desc += [g['product_short_description']['values'][0]['value'] if 'product_short_description' in g else '']
            long_desc += [g['product_long_description']['values'][0]['value'] if 'product_long_description' in g else '']
    
    return pd.DataFrame({'title':title, 'short_description':short_desc, 'long_description':long_desc})

def get_attribute_vals(wm_df_path, attr_key):
    values = []
    
    for wm_df in pd.read_csv(wm_df_path, sep='\t', encoding='utf-8', chunksize=5000):
        for pd_attr in wm_df.product_attributes:
            g = json.loads(pd_attr)
            if attr_key in g and 'values' in g[attr_key] and len(g[attr_key]['values']) > 0:
                v = []
                for j in range(len(g[attr_key]['values'])):
                    if 'value' in g[attr_key]['values'][j]:
                        v.append(g[attr_key]['values'][j]['value'])
                    else:
                        v.append('')
                        
                values.append('__'.join(v))
            else:
                values.append('')
    
    return values

def generate_wm_attr_value_counts(wm_df_path, wm_attr):
    wm_attr_value_counts = {}
    
    for wm_df in pd.read_csv(wm_df_path, sep='\t', encoding='utf-8', chunksize=5000):
        for pd_attr in wm_df.product_attributes:
            g = json.loads(pd_attr)

            for attr in set(wm_attr):
                if attr in g:
                    v = g[attr]

                    if 'values' in v and len(v['values']) > 0:
                        for j in range(len(v['values'])):
                            if 'value' in v['values'][j]:
                                x = ' '.join(utils.get_tokens(str(v['values'][j]['value']).lower())).strip()
                                x = re.sub('\s+', ' ', x)
                                if len(x) > 0:
                                    y = re.sub(r'([-+]?\d*\.\d+|\d+) fl oz', '\\1 oz', x)
                                    
                                    if x not in wm_attr_value_counts:
                                        wm_attr_value_counts[x] = 0
                                    wm_attr_value_counts[x] += 1
                                    
                                    if x != y:
                                        if y not in wm_attr_value_counts:
                                            wm_attr_value_counts[y] = 0
                                        wm_attr_value_counts[y] += 1
    return wm_attr_value_counts

def is_numeric_attribute(attr_vals):
    numeric = {}
    
    for attr, values in attr_vals.items():
        cnt = 0
        for val in values:
            numerics = re.findall(r"[-+]?\d*\.\d+|\d+", val, re.IGNORECASE)
            if len(numerics) > 0:
                cnt += 1
        
        if float(cnt)/len(values) > 0.5:
            numeric[attr] = True
    
    return numeric

def combine_msm_attr_values(msm_wm_map, msm_attr_vals, wm_attr_vals):
    combined_attr_vals = {}
        
    for w_attr, w_vals in wm_attr_vals.items():
        combined_attr_vals[w_attr] = w_vals
        w_attrs = w_attr.split('__')
        
        for w in w_attrs:
            if w in msm_attr_vals:
                combined_attr_vals[w_attr].update(msm_attr_vals[w])
        
        combined_attr_vals[w_attr] = [x for x in combined_attr_vals[w_attr] if len(x) > 2]

    return combined_attr_vals


def generate_wm_data_labels(wm_df_path, wm_attr, wm_msm_map):
    wm_sentences, wm_pcf_labels = [], []
    
    for wm_df in pd.read_csv(wm_df_path, sep='\t', encoding='utf-8', chunksize=5000):
        for pd_attr in wm_df.product_attributes:
            g = json.loads(pd_attr)

            title = g['product_name']['values'][0]['value'] if 'product_name' in g else ''
            short_desc = g['product_short_description']['values'][0]['value'] if 'product_short_description' in g else ''
            long_desc = g['product_long_description']['values'][0]['value'] if 'product_long_description' in g else ''

            corpus = title + " " + short_desc + " " + long_desc
            wm_sentences.append(corpus.lower())
            labels = {}

            for attr in set(wm_attr):
                u = []
                if attr in g:
                    v = g[attr]
                    if 'values' in v and len(v['values']) > 0:
                        for j in range(len(v['values'])):
                            if 'value' in v['values'][j]:
                                x = ' '.join(utils.get_tokens(str(v['values'][j]['value']).lower())).strip()
                                x = re.sub('\s+', ' ', x)
                                if len(x) > 0:
                                    y = re.sub(r'([-+]?\d*\.\d+|\d+) fl oz', '\\1 oz', x)
                                    u.append(x)
                                    if x != y:
                                        u.append(y)

                if attr in wm_msm_map:
                    labels[wm_msm_map[attr]] = u

            wm_pcf_labels.append(labels)
    
    return wm_sentences, wm_pcf_labels

def generate_wm_attr_values_normalized(wm_sentences, wm_pcf_labels, attribute_names, wm_attr_value_counts, min_count=10):
    indices = {}
    for i in range(len(wm_sentences)):
        attr_vals = wm_pcf_labels[i]
        for attr, values in attr_vals.items():
            if attr not in indices:
                indices[attr] = []
            indices[attr].append(i)
        
    wm_attr_values = {}
    
    for attr in attribute_names:
        if attr in indices:
            for i in indices[attr]:
                val = wm_pcf_labels[i][attr]
                
                if len(val) > 0:
                    if attr not in wm_attr_values:
                        wm_attr_values[attr] = set()
                    wm_attr_values[attr].update(val)
    
    numeric = is_numeric_attribute(wm_attr_values)
    
    for attr in attribute_names:
        if attr in indices:
            if attr in numeric and numeric[attr] is False:
                corpus, tags, filter_vals = [], [], set()

                for i in indices[attr]:
                    val = wm_pcf_labels[i][attr]
                    v_vals = []

                    for x in val:
                        if x in wm_attr_value_counts and wm_attr_value_counts[x] >= min_count:
                            filter_vals.add(x)
                            v_vals.append(x)

                    if len(v_vals) > 0:
                        corpus.append(wm_sentences[i])
                        tags.append(v_vals)

                wm_attr_values[attr] = filter_vals
                wm_attr_values[attr].update(utils.augment_ner_values_with_mi_features(corpus, tags, is_multi_label=True))
    
    return wm_attr_values

def generate_wm_attr_values_closed_list(wm_msm_map_df, wm_attr, wm_msm_map):
    wm_attr_values = {}
    
    for attr in set(wm_attr):
        df = wm_msm_map_df.loc[wm_msm_map_df['walmart_attribute'] == attr]
        if df.shape[0] > 0 and attr in wm_msm_map:
            attr = wm_msm_map[attr]
            
            attr_values = list(df.walmart_approved_values.apply(str).str.lower())
            attr_values = [x.split(';') for x in attr_values]
            attr_values = set(itertools.chain(*attr_values))
            
            wm_attr_values[attr] = set([' '.join(utils.get_tokens(x)) for x in attr_values if len(x) > 0])
    
    return wm_attr_values


def merge_walmart_values(closed_list_values, pcf_values):
    merged_values = set()
    
    for attr in closed_list_values:
        merged_values[attr] = closed_list_values[attr]
        if attr in pcf_values:
            merged_values[attr].update(pcf_values[attr])
    
    return merged_values


def fetch_attribute_map_data_mysql():
    mydb = mysql.connector.connect(host="localhost", user="yourusername", passwd="yourpassword", database="mydatabase")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM customers")
    
def pre_process_attr_values_for_tagging(combined_attr_vals_inverse):
    attr_vals_prefixes_word, attr_vals_prefixes_char = set(), set()
    
    for val in combined_attr_vals_inverse:
        val_tokens = utils.get_tokens(val)
        
        for j in range(1, len(val_tokens)+1):
            pref_token = ' '.join(val_tokens[:j])
            attr_vals_prefixes_word.add(pref_token)
        
        for j in range(1, len(val)+1):
            pref_token = ''.join(val[:j])
            attr_vals_prefixes_char.add(pref_token)
    
    return attr_vals_prefixes_word, attr_vals_prefixes_char

def merge_extractor_and_classifier_results(extraction_results, classification_results, attribute_names, u_classifier):
    pred_labels = [{} for i in range(len(extraction_results))]
    
    i = 0
    for x, y in zip(extraction_results, classification_results):
        for attr in attribute_names:
            if attr in x or attr in y:
                if attr in x:
                    pred_labels[i][attr] = x[attr]
                if attr in y and attr in u_classifier and u_classifier[attr]:
                    pred_labels[i][attr] += y[attr]
            else:
                pred_labels[i][attr] = ['None']
            
            pred_labels[i][attr] = list(set(pred_labels[i][attr]))
                
        i += 1
    
    return pred_labels


def process_boolean_attributes(pred_results, bool_attrs=set()):
    for i in range(len(pred_results)):
        result = pred_results[i]
        for attr, vals in result.items():
            if attr in bool_attrs:
                if len(vals) > 0 and len(vals[0]) > 0:
                    pred_results[i][attr] = ['Y']
                else:
                    pred_results[i][attr] = ['N']
    return pred_results


def fillback_with_pcf(pred_labels, pcf_labels, attribute_names):
    new_pred_labels = [{} for i in range(len(pred_labels))]
    
    for i in range(len(pred_labels)):
        attr_vals = pred_labels[i]
        for attr in attribute_names:
            if (attr not in attr_vals or len(attr_vals[attr]) == 0) and attr in pcf_labels[i]:
                new_pred_labels[i][attr] = pcf_labels[i][attr]
            else:
                if attr[:3].lower() != 'tov' and len(pred_labels[i][attr]) > 1 and attr in pcf_labels[i]:
                    new_pred_labels[i][attr] = list(set(pcf_labels[i][attr]).intersection(set(pred_labels[i][attr])))
                else:
                    new_pred_labels[i][attr] = pred_labels[i][attr]
    
    return new_pred_labels