import nltk, logging
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

#logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def getContents(type='train'):
    mydata = fetch_20newsgroups(subset=type, shuffle=True, random_state=42)

    contents = [" ".join(data.split("\n")) for data in mydata.data]
    labels = mydata.target

    return {'Contents':contents, 'Labels':labels}

def myTokenizer(text):
    return nltk.regexp_tokenize(text, "\\b[a-zA-Z]{3,}\\b")

def tokenizeContents(contents):
    return [myTokenizer(content) for content in contents]

def getVectorizer():
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', tokenizer=myTokenizer)

    return vectorizer