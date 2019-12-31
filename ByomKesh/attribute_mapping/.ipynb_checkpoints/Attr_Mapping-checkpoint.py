import pandas as pd
import numpy as np
import importlib, os, json, re, sys
import Utilities as utils
from sklearn.neighbors import BallTree
from nltk.stem import PorterStemmer
import logging, math
from feature_transformers.W2V_Transformer import W2VFeatures 

msm_file, wm_file = sys.argv[1], sys.argv[2]
msm_wm_map_file = sys.argv[3]

print("Reading data...")
df1 = pd.read_csv(msm_file)
df2 = pd.read_csv(wm_file, sep='\\t', engine='python')
df_map = pd.read_csv(msm_wm_map_file)
df_map.dropna(subset=['MSM_PF'], inplace=True)

print("Getting word vector model...")

if os.path.exists('word_vec_smart_sub.pkl') is False:
    sentences1 = list(df1.PFName.str.lower().apply(str) + " " + df1.AttrName.str.lower().apply(str) + " " + df1.AttrVal.str.lower().apply(str) + " " + df1.PDName.str.lower().apply(str) + " " + df1.PDDesc.str.lower().apply(str))

    sentences2 = []

    for i in range(len(df2.product_attributes)):
        g = json.loads(df2.product_attributes[i])
        
        title = g['product_name']['values'][0]['value'] if 'product_name' in g else ''
        short_desc = g['product_short_description']['values'][0]['value'] if 'product_short_description' in g else ''
        long_desc = g['product_long_description']['values'][0]['value'] if 'product_long_description' in g else ''

        corpus = title + " " + short_desc + " " + long_desc
        sentences2.append(str(corpus).lower().strip())

    wv = W2VFeatures(num_components=128, features=None, wv_type='W2V')
    wv.fit(sentences1 + sentences2)
    utils.save_data_pkl(wv, 'word_vec_smart_sub.pkl')
    
else:
    wv = utils.load_data_pkl('word_vec_smart_sub.pkl')
    
print("Reading excluded attributes...")
with open('excluded_attr_list.txt', 'r') as f:
    ex_attrs = f.readlines()
ex_attrs = set([x.strip() for x in ex_attrs])

print("Reading british to english dictionary...")
eng_df = pd.read_csv('british_to_american_english.csv')
eng_map = {}
brit = list(eng_df.BRITISH.str.lower().apply(str))
amer = list(eng_df.AMERICAN.str.lower().apply(str))
for i in range(len(brit)):
    brits = brit[i].split(",")
    amers = amer[i].split(",")
    for j in brits:
        j = j.strip()
        eng_map[j] = amers[0].strip()
            
msm_wm_map = {}

msm_pfs = list(df_map.MSM_PF)
wm_pts = list(df_map.WM_PT)

for i in range(len(msm_pfs)):
    if msm_pfs[i] not in msm_wm_map:
        msm_wm_map[msm_pfs[i]] = set()
    msm_wm_map[msm_pfs[i]].add(wm_pts[i])
    
df1 = df1.loc[df1['AttrVal'] != '']

for pf, wm_pts in msm_wm_map.items():
    df3 = df1.loc[df1['PFName'] == pf]
    
    if df3.shape[0] > 0:
        print("Creating attribute value maps...")
        attrs1 = list(set(df3.AttrName.apply(str)))
        attrs1_map = {}

        for attr in attrs1:
            df4 = df3.loc[df3['AttrName'] == attr]
            x = df4.AttrVal.str.lower().apply(str)
            x = [' '.join(utils.get_tokens(y)) for y in x]
            attrs1_map[attr.lower().strip()] = set(x)

        attrs2_map = {}
        for i in range(len(df2.product_attributes)):
            g = json.loads(df2.product_attributes[i])

            if 'product_type' in g and 'values' in g['product_type'] and len(g['product_type']['values']) > 0:
                if g['product_type']['values'][0]['value'] in wm_pts:
                    for k, v in g.items():
                        if k not in ex_attrs and 'values' in v and len(v['values']) > 0 and 'value' in v['values'][0]:
                            k = ' '.join(k.split('_'))
                            k = str(k).lower().strip()
                            k = ' '.join(utils.get_tokens(k))

                            if k not in attrs2_map:
                                attrs2_map[k] = set()
                            attrs2_map[k].add(str(v['values'][0]['value']).lower())

        print("Finding space break words...")
        val_break_words1, atr_break_words1 = {}, {}
        for k1, v1 in attrs1_map.items():
            val_break_words1[k1] = set()
            atr_break_words1[k1] = set()

            if len(k1) > 5:
                for i in range(3, len(k1)-3):
                    atr_break_words1[k1].add(k1[:i] + " " + k1[i:])

            for v in v1:
                if len(v) > 5:
                    for i in range(3, len(v)-3):
                        val_break_words1[k1].add(v[:i] + " " + v[i:])

        val_break_words2, atr_break_words2 = {}, {}
        for k2, v2 in attrs2_map.items():
            val_break_words2[k2] = set()
            atr_break_words2[k2] = set()

            if len(k2) > 5:
                for i in range(3, len(k2)-3):
                    atr_break_words2[k2].add(k2[:i] + " " + k2[i:])

            for v in v2:
                if len(v) > 5:
                    for i in range(3, len(v)-3):
                        val_break_words2[k2].add(v[:i] + " " + v[i:])
        
        print("Finding token sets...")
        tokenized_v1, tokenized_v1_trans = {}, {}
        tokenized_k1, tokenized_k1_trans = {}, {}
        
        for k1, v1 in attrs1_map.items():
            tokenized_k1[k1] = set(utils.get_tokens(k1))
            tokenized_k1_trans[k1] = set()
            
            for token in tokenized_k1_trans[k1]:
                if token in eng_map:
                    tokenized_k1_trans[k1].add(eng_map[token])
                else:
                    tokenized_k1_trans[k1].add(token)
            
            tokenized_v1[k1] = [set(utils.get_tokens(q1)) for q1 in v1]
            tokenized_v1_trans[k1] = []
            
            for token_set in tokenized_v1[k1]:
                q1_tokens = set()
                for token in token_set:
                    if token in eng_map:
                        q1_tokens.add(eng_map[token])
                    else:
                        q1_tokens.add(token)
                tokenized_v1_trans[k1].append(q1_tokens)
                
        tokenized_v2 = {}
        tokenized_k2 = {}
        
        for k2, v2 in attrs2_map.items():
            tokenized_k2[k2] = set(utils.get_tokens(k2))
            tokenized_v2[k2] = [set(utils.get_tokens(q2)) for q2 in v2]

        print("Finding matches...")            
        attr_name_match = {}

        print("Finding value-value match...")    
        for k1, v1 in attrs1_map.items():
            attr_name_match[k1] = {}
            v = []
            for k2, v2 in attrs2_map.items():
                y1 = v1.intersection(v2)
                
                y2 = set()
                if len(set(['yes', 'no', 'y', 'n']).intersection(v1)) == 0 and len(set(['yes', 'no', 'y', 'n']).intersection(v2)) == 0:
                    for p in tokenized_v1[k1] + tokenized_v1_trans[k1]:
                        for q in tokenized_v2[k2]:
                            r = min(len(p), len(q))
                            s = p.intersection(q)
                            if r > 0 and len(s) == r and abs(len(p)-len(q)) <= 2:
                                y2.add(' '.join(list(s)))

                y3 = val_break_words1[k1].intersection(v2)
                y4 = v1.intersection(val_break_words2[k2])

                if len(set(['yes', 'no', 'y', 'n']).intersection(y1)) == 0:
                    v.append((k2, len(y1)))

                if len(set(['yes', 'no', 'y', 'n']).intersection(y2)) == 0:
                    v.append((k2, len(y2)))

                if len(set(['yes', 'no', 'y', 'n']).intersection(y3)) == 0:
                    v.append((k2, len(y3)))

                if len(set(['yes', 'no', 'y', 'n']).intersection(y4)) == 0:
                    v.append((k2, len(y4)))


            if len(v) > 0:
                v = sorted(v, key=lambda k:-k[1])
                if v[0][1] > 0:
                    if v[0][0] not in attr_name_match[k1]:
                        attr_name_match[k1][v[0][0]] = 0
                    attr_name_match[k1][v[0][0]] += v[0][1]

                    for i in range(1, len(v)):
                        if v[i][1] == v[i-1][1]:
                            if v[i][0] not in attr_name_match[k1]:
                                attr_name_match[k1][v[i][0]] = 0
                            attr_name_match[k1][v[i][0]] += v[0][1]
                        else:
                            break
                            
        print(attr_name_match)

        print("Finding attr-value match...")    
        for k1, v1 in attrs1_map.items():
            for k2, v2 in attrs2_map.items():
                a1 = k1 in v2 or k2 in v1
                a2 = len(atr_break_words1[k1].intersection(v2)) > 0 or len(atr_break_words2[k2].intersection(v1)) > 0

                b = k1 in eng_map and len(set([eng_map[k1]]).intersection(v2)) > 0
                c = False

                p = tokenized_k1[k1]
                if len(set(['yes', 'no', 'y', 'n']).intersection(v2)) == 0:
                    for q in tokenized_v2[k2]:
                        r = min(len(p), len(q))
                        c = c or (r > 0 and len(p.intersection(q)) == r and abs(len(p)-len(q)) <=2)

                p = tokenized_k2[k2]
                if len(set(['yes', 'no', 'y', 'n']).intersection(v1)) == 0:
                    for q in tokenized_v1[k1]:
                        r = min(len(p), len(q))
                        c = c or (r > 0 and len(p.intersection(q)) == r and abs(len(p)-len(q)) <=2)
                        
                d = False
                
                p = tokenized_k1_trans[k1]
                if len(set(['yes', 'no', 'y', 'n']).intersection(v2)) == 0:
                    for q in tokenized_v2[k2]:
                        r = min(len(p), len(q))
                        d = d or (r > 0 and len(p.intersection(q)) == r and abs(len(p)-len(q)) <=2)

                p = tokenized_k2[k2]
                if len(set(['yes', 'no', 'y', 'n']).intersection(v1)) == 0:
                    for q in tokenized_v1_trans[k1]:
                        r = min(len(p), len(q))
                        d = d or (r > 0 and len(p.intersection(q)) == r and abs(len(p)-len(q)) <=2)

                if a1 or a2 or b or c or d:
                    if k2 not in attr_name_match[k1]:
                        attr_name_match[k1][k2] = 0
                    attr_name_match[k1][k2] += 1
                    
        print(attr_name_match)
        

        print("Finding attr-attr match...")    
        for k1, v1 in attrs1_map.items():
            for k2, v2 in attrs2_map.items():
                a1 = k1 == k2
                a2 = k1 in atr_break_words2[k2] or k2 in atr_break_words1[k1]

                b = k1 in eng_map and k2 in eng_map[k1]

                p, q = set(utils.get_tokens(k1)), set(utils.get_tokens(k2))
                r = min(len(p), len(q))
                c = r > 0 and len(p.intersection(q)) == r and abs(len(p)-len(q)) <=2

                w = []
                for p1 in p:
                    if p1 in eng_map:
                        w += [list(eng_map[p1])[0]]
                    else:
                        w += [p1]

                w = ' '.join(w)
                d = w != k1 and w == k2

                p, q = set(utils.get_tokens(w)), set(utils.get_tokens(k2))
                r = min(len(p), len(q))
                e = w != k1 and r > 0 and len(p.intersection(q)) == r and abs(len(p)-len(q)) <=2

                if a1 or a2 or b or c or d or e:
                    if k2 not in attr_name_match[k1]:
                        attr_name_match[k1][k2] = 0
                    attr_name_match[k1][k2] += 1

        print(attr_name_match)

        print("Finding matches word vector...")                
        attr_val_vecs1, attr_vecs1 = {}, {}

        for k1, v1 in attrs1_map.items():
            v1 = [x for x in v1 if len(list(x)) > 0 and len(re.findall(r'[A-Za-z ]+', x)) > 0]
            attr_vecs1[k1] = wv.transform([k1])[0]

            if len(v1) > 0:
                vecs = wv.transform(v1)
                attr_val_vecs1[k1] = np.mean(vecs, axis=0)
            else:
                attr_val_vecs1[k1] = np.zeros(128)

        attr_val_vecs2, attr_vecs2 = {}, {}
        for k2, v2 in attrs2_map.items():
            v2 = [x for x in v2 if len(list(x)) > 0 and len(re.findall(r'[A-Za-z ]+', x)) > 0]
            attr_vecs2[k2] = wv.transform([k2])[0]

            if len(v2) > 0:
                vecs = wv.transform(v2)
                attr_val_vecs2[k2] = np.mean(vecs, axis=0)
            else:
                attr_val_vecs2[k2] = np.zeros(128)

        for k1, v1 in attrs1_map.items():
            v = []
            for k2, v2 in attrs2_map.items():

                d1 = np.abs(attr_val_vecs1[k1]-attr_val_vecs2[k2]) if np.sum(attr_val_vecs1[k1]) != 0 and np.sum(attr_val_vecs2[k2]) != 0 else float("Inf")
                d2 = np.abs(attr_vecs1[k1]-attr_val_vecs2[k2]) if np.sum(attr_vecs1[k1]) != 0 and np.sum(attr_val_vecs2[k2]) != 0 else float("Inf")
                d3 = np.abs(attr_val_vecs1[k1]-attr_vecs2[k2]) if np.sum(attr_val_vecs1[k1]) != 0 and np.sum(attr_vecs2[k2]) != 0 else float("Inf")
                d4 = np.abs(attr_vecs1[k1]-attr_vecs2[k2]) if np.sum(attr_vecs1[k1]) != 0 and np.sum(attr_vecs2[k2]) != 0 else float("Inf")

                d1 = np.sum(d1**2)
                d2 = np.sum(d2**2)
                d3 = np.sum(d3**2)
                d4 = np.sum(d4**2)

                if len(set(['yes', 'no', 'y', 'n']).intersection(v2)) == 0 or len(set(['yes', 'no', 'y', 'n']).intersection(v1)) == 0:
                    d = min(d1, d2, d3, d4)
                else:
                    d = min(d2, d3, d4)

                v.append((k2, d))

            v = sorted(v, key=lambda k:k[1])

            if v[0][0] not in attr_name_match[k1]:
                attr_name_match[k1][v[0][0]] = 0
            attr_name_match[k1][v[0][0]] += 1

            for i in range(1, len(v)):
                if v[i][1] == v[i-1][1]:
                    if v[i][0] not in attr_name_match[k1]:
                        attr_name_match[k1][v[i][0]] = 0
                    attr_name_match[k1][v[i][0]] += 1
                else:
                    break

        print(attr_name_match)

        msm_attr, wm_attr, score, msm_values, wm_values = [], [], [], [], []
        for k, v in attr_name_match.items():
            msm_attr += [k]*len(v)
            wm_attr += ['_'.join(x.split()) for x, y in v.items()]
            score += [y for x, y in v.items()]
            msm_values += [', '.join(list(attrs1_map[k]))]*len(v)
            wm_values += [', '.join(list(attrs2_map[x])) for x, y in v.items()]

        df = pd.DataFrame({'msm_attribute':msm_attr, 'wm_attribute':wm_attr, 'score':score, 'msm_values':msm_values, 'wm_values':wm_values})
        df.to_csv(os.path.join('attr_mapping_folder6', '_'.join(str(pf).split()) + '_mappings.csv'))