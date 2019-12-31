import sys
sys.path.append('/home/jupyter/MySuperMarket')

import pandas as pd
import numpy as np
import os, re, math, json, nltk, ast
import Utilities as utils
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import attribute_extraction.PreProcessingUtils as pputils
from attribute_extraction.CRF_Extractor import CRFExtractor
from attribute_extraction.Classifier import Classifier

df1 = pd.read_csv('data/ui_mapping.csv')

m_attr_curr = 'TOV Flavour'

gpt_pt_map = {}
pts = list(df1.walmart_pcf_product_type)
gpts = list(df1.walmart_global_product_type)

for i in range(len(pts)):
    gpt_pt_map[gpts[i]] = pts[i]
    
with open('data/iron_bank_flavors.txt', encoding='utf-8') as f:
    ib_flavors = f.readlines()

ib_flavors = set([x.strip().lower() for x in ib_flavors])

gpts, pts, p, r, s, c, t, t_pcf, t_mod, pcf_fail, model_gen, pcf_no_closed_list = [], [], [], [], [], [], [], [], [], [], [], []

for gpt in set(df1.walmart_global_product_type):
    pt = gpt_pt_map[gpt]
    
    print("Current GPT : ", gpt)
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
        
        print("Creating MSM attribute type map...")
        msm_type_map = pputils.generate_msm_type_map(msm_attr, msm_attr_type)

        print("Creating MSM attributes to values map...")
        msm_attr_vals = pputils.generate_msm_attr_values_closed_list(df_new, msm_attr, msm_type_map)
        
        if m_attr_curr in msm_attr_vals:
            msm_attr_vals = msm_attr_vals[m_attr_curr]

            w_attr_curr = msm_wm_map[m_attr_curr]

            print("Getting OG dataframe...")
            df2_new_pred_path = os.path.join('data', 'product_attributes_og2', f_name + '.tsv')

            print("Getting GPTs...")
            gpt_list = pputils.get_attribute_vals(df2_new_pred_path, 'global_product_type')

            attr_val_list = []

            for w_attr_x in w_attr_curr:
                print("Getting true values from PCF...")
                curr_attr_val_list = pputils.get_attribute_vals(df2_new_pred_path, w_attr_x)

                if len(curr_attr_val_list) > 0:
                    for i in range(len(curr_attr_val_list)):
                        y = list(set(curr_attr_val_list[i].lower().split('__')))
                        z = []
                        for y1 in y:
                            if len(y1) > 0:
                                y1 = re.sub('fl oz', 'oz', y1)
#                             if len(y1) > 0 and (y1 in msm_attr_vals or y1 in ib_flavors):
                                z.append(y1)
                        curr_attr_val_list[i] = z

                    filtered_attr_val_list = []
                    for i in range(len(gpt_list)):
                        if gpt_list[i] == gpt:
                            filtered_attr_val_list.append(curr_attr_val_list[i])

                    if len(attr_val_list) == 0:
                        attr_val_list = [filtered_attr_val_list[i] for i in range(len(filtered_attr_val_list))]
                    else:
                        attr_val_list = [attr_val_list[i] + filtered_attr_val_list[i] for i in range(len(filtered_attr_val_list))]
                        
            attr_val_list = [list(set(x)) for x in attr_val_list]
            
            
            pcf_attr_val_list = []

            for w_attr_x in w_attr_curr:
                print("Getting true values from PCF...")
                curr_attr_val_list = pputils.get_attribute_vals(df2_new_pred_path, w_attr_x)

                if len(curr_attr_val_list) > 0:
                    for i in range(len(curr_attr_val_list)):
                        y = list(set(curr_attr_val_list[i].lower().split('__')))
                        z = []
                        for y1 in y:
                            if len(y1) > 0:
                                y1 = re.sub('fl oz', 'oz', y1)
                                z.append(y1)
                        curr_attr_val_list[i] = z

                    filtered_attr_val_list = []
                    for i in range(len(gpt_list)):
                        if gpt_list[i] == gpt:
                            filtered_attr_val_list.append(curr_attr_val_list[i])

                    if len(pcf_attr_val_list) == 0:
                        pcf_attr_val_list = [filtered_attr_val_list[i] for i in range(len(filtered_attr_val_list))]
                    else:
                        pcf_attr_val_list = [pcf_attr_val_list[i] + filtered_attr_val_list[i] for i in range(len(filtered_attr_val_list))]
                        
            pcf_attr_val_list = [list(set(x)) for x in pcf_attr_val_list] 

            if len(attr_val_list) > 0:
                print("Getting predicted brands...")
                df_out = pd.read_csv(os.path.join('outputs', '_'.join(pt.split(' ')) + '_extracted_no_pcf.csv'))
                df_out = df_out.loc[df_out['global_product_type'] == gpt]

                if m_attr_curr in df_out.columns:
                    if m_attr_curr == 'Size':
                        pred_attr_vals = []
                        descs = list(df_out['wm_desc'])
                        for desc in descs:
                            vals = re.findall(r'([-+]?\d*\.\d+|\d+)\s*(fl|fl\.)?\s*(oz)', desc, re.IGNORECASE)
                            vals = [re.sub('\s+', ' ', ' '.join(x)) for x in vals]
                            pred_attr_vals.append(vals)
                            
                    else:
                        pred_attr_vals = list(df_out[m_attr_curr])
                        pred_attr_vals = [ast.literal_eval(x) for x in pred_attr_vals]

                        for i in range(len(pred_attr_vals)):
                            y = pred_attr_vals[i]
                            z = []
                            for y1 in y:
                                if len(y1) > 0:
    #                             if len(y1) > 0 and (y1 in msm_attr_vals or y1 in ib_flavors):
                                    z.append(y1)
                            pred_attr_vals[i] = z

                    pred_attr_vals = [list(set(x)) for x in pred_attr_vals]
        
                    print("Getting classification score...")
                    precision, recall, f_score, support = utils.custom_classification_scores(attr_val_list, pred_attr_vals)

                    print("Getting coverage...")
                    cnt, corr = 0, 0
                    for i in range(len(pred_attr_vals)):
                        a1 = i < len(attr_val_list) and ((len(attr_val_list[i]) == 1 and len(attr_val_list[i][0]) == 0) or len(attr_val_list[i]) == 0)
                        b1 = (len(pred_attr_vals[i]) == 1 and len(pred_attr_vals[i][0]) != 0) or len(pred_attr_vals[i]) > 1
                        
                        if a1 and b1:
                            corr += 1
                            
                    for i in range(len(attr_val_list)):
                        if (len(attr_val_list[i]) == 1 and len(attr_val_list[i][0]) == 0) or len(attr_val_list[i]) == 0:
                            cnt += 1

                    total = len(attr_val_list)
                    total_pcf = len([x for x in attr_val_list if ((len(x) == 1 and len(x[0]) != 0) or len(x) > 1)])
                    total_model = len([x for x in pred_attr_vals if ((len(x) == 1 and len(x[0]) != 0) or len(x) > 1)])
                    coverage = float(corr)/total if total != 0 else float('NaN')
                    
                    pcf_closed_list = 0
                    for i in range(len(pcf_attr_val_list)):
                        a1 = (len(pcf_attr_val_list[i]) == 1 and len(pcf_attr_val_list[i][0]) != 0) or len(pcf_attr_val_list[i]) > 1
                        b1 = (len(attr_val_list[i]) == 1 and len(attr_val_list[i][0]) == 0) or len(attr_val_list[i]) == 0
                        
                        if a1 and b1:
                            pcf_closed_list += 1

                    gpts.append(gpt)
                    pts.append(pt)
                    p.append(precision)
                    r.append(recall)
                    s.append(support)
                    c.append(coverage)
                    t.append(total)
                    t_pcf.append(total_pcf)
                    t_mod.append(total_model)
                    pcf_fail.append(cnt)
                    model_gen.append(corr)
                    pcf_no_closed_list.append(pcf_closed_list)
            
out_df = pd.DataFrame({'GPT':gpts, 'PT':pts, 'precision':p, 'recall':r, 'support':s, 'incremental_coverage':c, 'total_sku':t, 'total_sku_pcf_val':t_pcf, 'total_sku_model_val':t_mod, 'pcf_missing_cnt':pcf_fail, 'model_pred_pcf_missing_cnt':model_gen, 'pcf_val_not_closed_list':pcf_no_closed_list})
out_df.to_csv('results.csv', sep=',')
            
        
        
        