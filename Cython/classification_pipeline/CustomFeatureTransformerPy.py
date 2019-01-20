from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import numpy as np, random, re, math
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def sort_key(key):
    return -key[1]

def clean_tokens(tokens, to_replace='[^\w\-]+'):
    lemma = WordNetLemmatizer()
    
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    tokens = [lemma.lemmatize(token) for token in tokens]
    
    return tokens

def tokenize(mystr):
    tokenizer = RegexpTokenizer('[^ ]+')
    mystr = mystr.lower()

    return tokenizer.tokenize(mystr)

def get_tokens(sentence, to_replace='[^\w\-]+'):
    sentence = re.sub('<[^<]+?>', ' ', sentence)
    sentence = re.sub(to_replace, ' ', sentence)
    
    tokens = clean_tokens(tokenize(sentence), to_replace)
    tokens = [token.strip() for token in tokens]
    
    return tokens

def get_features_mi(sentences, labels, num_feats=100, is_multi_label=True):
    a, b, c = defaultdict(float), defaultdict(float), defaultdict(float)

    total = 0

    for idx in range(len(sentences)):
        sent = sentences[idx]
        label = labels[idx]
        
        common_tokens = set(get_tokens(sent))
        
        total += 1
        
        for token in common_tokens:
            a[token] += 1
            
        if is_multi_label:
            for color in label:
                b[color] += 1
        else:
            b[label] += 1
            
        for token in common_tokens:
            if is_multi_label:
                for color in label:
                    c[(color, token)] += 1
            else:
                c[(label, token)] += 1
    
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

        a1 = p*np.log2(float(p)/(u*w)) if p != 0 else 0
        a2 = q*np.log2(float(q)/(u*z)) if q != 0 else 0
        a3 = r*np.log2(float(r)/(v*w)) if r != 0 else 0
        a4 = s*np.log2(float(s)/(v*z)) if s != 0 else 0

        mi = a1 + a2 + a3 + a4
        
        mi_values[token] = max(mi_values[token], mi)
    
    final_tokens = [(token, val) for token, val in mi_values.items()]
    final_tokens = sorted(final_tokens, key=sort_key)[:min(num_feats, len(final_tokens))]
    
    final_tokens = [x for x, y in final_tokens]
    
    return final_tokens


class AreaRugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_features=100, min_ngram=1, max_ngram=3):
        self.vectorizer = None
        self.num_features = num_features
        self.min_ngram = max(1, min_ngram)
        self.max_ngram = max(min_ngram, max_ngram)

    def fit(self, X, y):
        features = get_features_mi(X, y, self.num_features)
        
        self.vectorizer = TfidfVectorizer(tokenizer=get_tokens, ngram_range=(self.min_ngram, self.max_ngram), stop_words='english', binary=True, vocabulary=features)
        self.vectorizer.fit(X)
        
    def transform(self, X):
        return self.vectorizer.transform(X)