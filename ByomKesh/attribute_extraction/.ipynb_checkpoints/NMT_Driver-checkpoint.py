from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np, pandas as pd
import os
import io
import time
from attribute_extraction.Neural_Machine_Translation import NMTNetwork
import Utilities as utils

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang, use_char=False):
    if use_char:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True, oov_token='!')
    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='UNK')
        
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    return tensor, lang_tokenizer

def load_dataset(src, trg, use_char=False):
    src_tensor, src_tokenizer = tokenize(src, use_char=use_char)
    trg_tensor, trg_tokenizer = tokenize(trg, use_char=use_char)

    return src_tensor, trg_tensor, src_tokenizer, trg_tokenizer


def get_data_as_generator(src_tensor, trg_tensor, batch_size=64):
    n = len(src_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((src_tensor, trg_tensor)).shuffle(n)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    return iter(dataset)

class NMTDriver():
    def __init__(self, use_char=False):
        self.use_char = use_char
        self.nmt = {}
        self.start_token, self.end_token = chr(0), chr(1)
        self.src_tokenizer, self.trg_tokenizer = None, {}
        self.max_length_src, self.max_length_trg = None, {}
        
    
    def train(self, src_sents, trg_sents_attrs):
        src_sents = utils.pre_process_nmt(src_sents, self.start_token, self.end_token)
        
        trg_sents_map = {}
        for i in range(len(trg_sents_attrs)):
            for attr, trg_sent in trg_sents_attrs[i].items():
                if attr not in trg_sents_map:
                    trg_sents_map[attr] = []
                trg_sents_map[attr].append(trg_sent)
        
        for attr, trg_sents in trg_sents_map.items():
            trg_sents = utils.pre_process_nmt(trg_sents, self.start_token, self.end_token)

            print("Pre-processing NMT tensors...")
            src_tensor, trg_tensor, self.src_tokenizer, self.trg_tokenizer[attr] = load_dataset(src_sents, trg_sents, self.use_char)
            self.max_length_trg[attr], self.max_length_src = max_length(trg_tensor), max_length(src_tensor)

            src_tensor = tf.keras.preprocessing.sequence.pad_sequences(src_tensor, maxlen=self.max_length_src, padding='post')
            trg_tensor = tf.keras.preprocessing.sequence.pad_sequences(trg_tensor, maxlen=self.max_length_trg[attr], padding='post')

            print("Training NMT...")
            self.nmt[attr] = NMTNetwork(get_data_as_generator, src_tensor, trg_tensor, self.src_tokenizer, self.trg_tokenizer[attr], 
                                  self.max_length_src, self.max_length_trg[attr], self.start_token, self.end_token)
            self.nmt[attr].fit()
        
    
    def predict(self, src_sents):
        print("Pre-processing NMT tensors...")
        n = len(src_sents)
        batch_size = 1000
        
        num_batches = int((n+batch_size-1)/batch_size)
        
        print("Predicting NMT...")
        outputs = [{} for i in range(n)]
        
        separator = '' if self.use_char else ' '
        
        for m in range(num_batches):
            start, end = m*batch_size, min((m+1)*batch_size, n)
            
            src_sents_batch = [src_sents[x] for x in range(start, end)]
            src_sents_batch = utils.pre_process_nmt(src_sents_batch, self.start_token, self.end_token)
        
            src_tensor = self.src_tokenizer.texts_to_sequences(src_sents_batch)
            src_tensor = tf.keras.preprocessing.sequence.pad_sequences(src_tensor, maxlen=self.max_length_src, padding='post')
            src_tensor = tf.convert_to_tensor(src_tensor)
            
            for attr, nmt in self.nmt.items():
                out = nmt.predict(src_tensor, separator)
                for i in range(len(out)):
                    outputs[start+i][attr] = out[i]
        
        return outputs

