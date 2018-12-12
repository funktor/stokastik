from collections import defaultdict
import numpy as np, random, re, math
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
import pandas as pd

def clean_tokens(tokens, to_replace='[^\w\-\+\&\.\'\"]+'):
    lemma = WordNetLemmatizer()
    
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    tokens = [lemma.lemmatize(token) for token in tokens]
    
    return tokens

def tokenize(mystr):
    tokenizer = RegexpTokenizer('[^ ]+')
    mystr = mystr.lower()

    return tokenizer.tokenize(mystr)

def get_tokens(sentence, to_replace='[^\w\-\+\&\.\'\"]+'):
    sentence = re.sub('<[^<]+?>', ' ', str(sentence))
    sentence = re.sub(to_replace, ' ', str(sentence))
    
    tokens = clean_tokens(tokenize(sentence), to_replace)
    tokens = [token.strip() for token in tokens]
    
    return tokens

def pca_reduction(vectors):
    pca = TruncatedSVD(n_components=128).fit(vectors)
    return pca.fit_transform(vectors), pca

def get_unique_items_pt(product_data, pt):
    items = [x for x in product_data if x[1] == pt]
    
    unique_items, added_item_ids = [], set()

    for item in items:
        if item[0] not in added_item_ids:
            unique_items.append(item)
            added_item_ids.add(item[0])
    
    return unique_items

def get_text_data(product_data_items):
    titles = [str(x[2]) for x in product_data_items]
    return titles

def string_diff(str1, str2):
    cached = defaultdict(dict)
    tokens1, tokens2 = get_tokens(str1), get_tokens(str2)
    
    for i in range(-1, len(tokens1)):
        for j in range(-1, len(tokens2)):
            if i == -1 or j == -1:
                cached[i][j] = [[]]
            else:
                if tokens1[i] == tokens2[j]:
                    out = [x + [(tokens1[i], i)] for x in cached[i - 1][j - 1]]
                else:
                    a, b = cached[i - 1][j], cached[i][j - 1]
                    if len(a[0]) == len(b[0]):
                        out = a + b if a[len(a) - 1] != b[len(b) - 1] else a
                    else:
                        out = a if len(a[0]) > len(b[0]) else b
                        
                cached[i][j] = out
                
    longest = cached[len(tokens1) - 1][len(tokens2) - 1][0]
    
    diff, start = [], 0
    
    for _, pos in longest:
        if start < pos:
            diff.append(tokens1[start:pos])
            
        start = pos + 1
    
    diff.append(tokens1[start:])
    
    return diff