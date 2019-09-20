from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np, pandas as pd
import os
import io
import time

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
#     w = unicode_to_ascii(w.lower().strip())
    
    w = w.lower().strip()
    
    w = re.sub('<[^<]+?>', ' ', w)
    w = re.sub('[^\w\-\+\&\.\'\"\:]+', ' ', w)

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

#     w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    df = pd.read_csv(path, nrows=num_examples)
        
    titles = [preprocess_sentence(x) for x in list(df['title'])]
    desc = [preprocess_sentence(x) for x in list(df['short_description'])]
    
    return titles, desc
        
        
#     lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
#     word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

#     return zip(*word_pairs)

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=100, padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer