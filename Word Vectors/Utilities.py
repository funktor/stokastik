from glob import glob
import os, re, nltk, logging
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def getContents(type='train'):
    mydata = fetch_20newsgroups(subset=type, shuffle=True, random_state=42)

    contents = [" ".join(data.split("\n")) for data in mydata.data]
    labels = mydata.target

    return {'Contents':contents, 'Labels':labels}

def tokenizeContents(contents):
    return [nltk.regexp_tokenize(content, "\\b[a-zA-Z]{3,}\\b") for content in contents]

def getVectorizer():
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english',
                                 token_pattern='\\b[a-zA-Z]{3,}\\b', tokenizer=nltk.word_tokenize)

    return vectorizer