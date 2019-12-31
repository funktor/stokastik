import pandas as pd
import numpy as np
import os, re, math, json, nltk, copy
import Utilities as utils
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from attribute_extraction.SuffixTree import SuffixTree

def pre_process_attr_values_for_tagging(attr_values_inverse_map):
    attr_vals_prefixes_word, attr_vals_prefixes_char = set(), set()
    
    for val in attr_values_inverse_map:
        val_tokens = utils.get_tokens(val)

        for j in range(1, len(val_tokens)+1):
            pref_token = ' '.join(val_tokens[:j])
            attr_vals_prefixes_word.add(pref_token)

        for j in range(1, len(val)+1):
            pref_token = ''.join(val[:j])
            attr_vals_prefixes_char.add(pref_token)
    
    return attr_vals_prefixes_word, attr_vals_prefixes_char

def get_attr_values_inverse_map(attr_values_map, is_multi_label=True):
    attr_vals_inverse_map, temp = {}, {}

    for attr, values in attr_values_map.items():
        if is_multi_label:
            for val in values:
                val = ' '.join(utils.get_tokens(val))
                if val not in temp:
                    temp[val] = set()
                temp[val].add(attr)
        else:
            val = ' '.join(utils.get_tokens(values))
            if val not in temp:
                temp[val] = set()
            temp[val].add(attr)

    for val, attrs in temp.items():
        if len(attrs) == 1:
            attr_vals_inverse_map[val] = list(attrs)[0]
    
    return attr_vals_inverse_map

def get_tokenized_values(attr_values_inverse_map):
    value_tokenized_word, value_tokenized_char = [], []
        
    attr_values = [(val, attr) for val, attr in attr_values_inverse_map.items()]
    attr_values = sorted(attr_values, key=lambda k:-len(k[0]))

    for i in range(len(attr_values)):
        val_word_tokens = utils.get_tokens(attr_values[i][0])
        val_char_tokens = utils.get_tokens(attr_values[i][0], is_char=True)

        value_tokenized_word.append(val_word_tokens)
        value_tokenized_char.append(val_char_tokens)
        
    return attr_values, value_tokenized_word, value_tokenized_char

class BIOEncoder():
    def __init__(self, max_word_length=150, max_char_length=500, attr_values_map=None, is_multi_label=False):
        self.is_multi_label = is_multi_label
        self.attr_values_map = attr_values_map
        self.max_word_length = max_word_length
        self.max_char_length = max_char_length
        
    
    def bioe_encoding_suffix_tree(self, sentences, sent_pcf_labels=None, mode='Train'):
        print("Doing BIOE Encoding...")
        
        bioe_words, bioe_chars = [], []
        global_attr_values_inverse_map = {}
        global_value_tokenized_word, global_value_tokenized_char, global_attr_values = [], [], []
        
        if self.attr_values_map is not None:
            global_attr_values_inverse_map = get_attr_values_inverse_map(self.attr_values_map)
            global_attr_values, global_value_tokenized_word, global_value_tokenized_char = get_tokenized_values(global_attr_values_inverse_map)
        
        for i in range(len(sentences)):
            corpus = sentences[i]

            word_tokens = utils.get_tokens(corpus, min_ngram=1, max_ngram=1)
            char_tokens = utils.get_tokens(corpus, min_ngram=1, max_ngram=1, is_char=True)
            
            word_tokens = word_tokens[:min(len(word_tokens), self.max_word_length)]
            char_tokens = char_tokens[:min(len(char_tokens), self.max_char_length)]

            word_tokens = [w for w in word_tokens if len(w) > 0]
            char_tokens = [w for w in char_tokens if len(w) > 0]

            word_pos_tags = [y for x, y in nltk.pos_tag(word_tokens)]
            char_pos_tags = ['UNK']*len(char_tokens)

            word_ner_tags = ['O']*len(word_tokens)
            char_ner_tags = ['O']*len(char_tokens)
            
            if mode == 'Train':
                word_suffix_tree = SuffixTree()
                char_suffix_tree = SuffixTree()

                word_suffix_tree.build_tree(word_tokens)
                char_suffix_tree.build_tree(char_tokens)
                
                attr_values_inverse_map = {}
                attr_values, value_tokenized_word, value_tokenized_char = [], [], []
                
                if sent_pcf_labels is not None and len(sent_pcf_labels[i]) > 0:
                    attr_values_inverse_map = get_attr_values_inverse_map(sent_pcf_labels[i], self.is_multi_label)
                    attr_values, value_tokenized_word, value_tokenized_char = get_tokenized_values(attr_values_inverse_map)
                
                if self.attr_values_map is not None:
                    attr_values_inverse_map.update(global_attr_values_inverse_map)
                    attr_values += global_attr_values
                    value_tokenized_word += global_value_tokenized_word
                    value_tokenized_char += global_value_tokenized_char

                for j in range(len(attr_values)):
                    val, attr = attr_values[j]

                    val_word_tokens = value_tokenized_word[j]
                    val_char_tokens = value_tokenized_char[j]

                    word_pos = word_suffix_tree.search(val_word_tokens)
                    char_pos = char_suffix_tree.search(val_char_tokens)

                    length = len(val_word_tokens)

                    if word_pos is not None:
                        for k in word_pos:
                            if length == 1 and word_ner_tags[k] == 'O':
                                word_ner_tags[k] = 'S-' + attr

                            elif length == 2 and word_ner_tags[k] == 'O' and word_ner_tags[k+1] == 'O':
                                word_ner_tags[k] = 'B-' + attr
                                word_ner_tags[k+1] = 'E-' + attr

                            elif word_ner_tags[k] == 'O' and word_ner_tags[k+length-1] == 'O':
                                word_ner_tags[k] = 'B-' + attr
                                word_ner_tags[k+length-1] = 'E-' + attr
                                word_ner_tags[j+1:k+length-1] = ['I-' + attr]*(k+length-2)

                    length = len(val_char_tokens)

                    if char_pos is not None:
                        for k in char_pos:
                            if length == 1 and char_ner_tags[k] == 'O':
                                char_ner_tags[k] = 'S-' + attr

                            elif length == 2 and char_ner_tags[k] == 'O' and char_ner_tags[k+1] == 'O':
                                char_ner_tags[k] = 'B-' + attr
                                char_ner_tags[k+1] = 'E-' + attr

                            elif char_ner_tags[k] == 'O' and char_ner_tags[k+length-1] == 'O':
                                char_ner_tags[k] = 'B-' + attr
                                char_ner_tags[k+length-1] = 'E-' + attr
                                char_ner_tags[j+1:k+length-1] = ['I-' + attr]*(k+length-2)

            q_words = zip(word_tokens, word_pos_tags, word_ner_tags)
            q_chars = zip(char_tokens, char_pos_tags, char_ner_tags)

            bioe_words.append(list(q_words))
            bioe_chars.append(list(q_chars))
            
        return bioe_words, bioe_chars
    
        
    def bioe_encoding(self, sentences, sent_pcf_labels=None, mode='Train'):
        print("Doing BIOE Encoding...")
        bioe_words, bioe_chars = [], []
        
        if self.attr_values_map is not None:
            global_attr_values_inverse_map = get_attr_values_inverse_map(self.attr_values_map)
            global_attr_vals_prefixes_word, global_attr_vals_prefixes_char = pre_process_attr_values_for_tagging(global_attr_values_inverse_map)
        
        for i in range(len(sentences)):
            corpus = sentences[i]

            word_tokens = utils.get_tokens(corpus, min_ngram=1, max_ngram=1)
            char_tokens = utils.get_tokens(corpus, min_ngram=1, max_ngram=1, is_char=True)
            
            word_tokens = word_tokens[:min(len(word_tokens), self.max_word_length)]
            char_tokens = char_tokens[:min(len(char_tokens), self.max_char_length)]

            word_tokens = [w for w in word_tokens if len(w) > 0]
            char_tokens = [w for w in char_tokens if len(w) > 0]

            word_pos_tags = [y for x, y in nltk.pos_tag(word_tokens)]
            char_pos_tags = ['UNK']*len(char_tokens)

            word_ner_tags = ['O']*len(word_tokens)
            char_ner_tags = ['O']*len(char_tokens)
            
            if mode == 'Train':
                attr_values_inverse_map = {}
                attr_vals_prefixes_word, attr_vals_prefixes_char = set(), set()
                
                if sent_pcf_labels is not None and len(sent_pcf_labels[i]) > 0:
                    attr_values_inverse_map = get_attr_values_inverse_map(sent_pcf_labels[i], self.is_multi_label)
                    attr_vals_prefixes_word, attr_vals_prefixes_char = pre_process_attr_values_for_tagging(attr_values_inverse_map)
                    
                if self.attr_values_map is not None:
                    attr_values_inverse_map.update(global_attr_values_inverse_map)
                    attr_vals_prefixes_word.update(global_attr_vals_prefixes_word)
                    attr_vals_prefixes_char.update(global_attr_vals_prefixes_char)
                
                j = 0
                while j < len(word_tokens):
                    k, attr = j, None
                    while k < len(word_tokens):
                        sub_word_token = ' '.join(word_tokens[j:k+1])

                        if sub_word_token in attr_vals_prefixes_word:
                            if sub_word_token in attr_values_inverse_map:
                                attr = attr_values_inverse_map[sub_word_token]
                        else:
                            break

                        k += 1

                    if attr is not None:
                        length = k-j

                        if length == 1 and word_ner_tags[j] == 'O':
                            word_ner_tags[j] = 'S-' + attr
                            
                        elif length == 2 and word_ner_tags[j:j+2] == ['O']*2:
                            word_ner_tags[j] = 'B-' + attr
                            word_ner_tags[j+1] = 'E-' + attr
                            
                        elif length > 2 and word_ner_tags[j:k] == ['O']*length:
                            word_ner_tags[j] = 'B-' + attr
                            word_ner_tags[k-1] = 'E-' + attr
                            word_ner_tags[j+1:k-1] = ['I-' + attr]*(length-2)
                            
                        j = k
                    else:
                        j += 1

                j = 0
                while j < len(char_tokens):
                    k, attr = j, None
                    while k < len(char_tokens):
                        sub_char_token = ''.join(char_tokens[j:k+1])

                        if sub_char_token in attr_vals_prefixes_char:
                            if sub_char_token in attr_values_inverse_map:
                                attr = attr_values_inverse_map[sub_char_token]
                        else:
                            break

                        k += 1

                    if attr is not None:
                        length = k-j

                        if length == 1 and char_ner_tags[j] == 'O':
                            char_ner_tags[j] = 'S-' + attr
                            
                        elif length == 2 and char_ner_tags[j:j+2] == ['O']*2:
                            char_ner_tags[j] = 'B-' + attr
                            char_ner_tags[j+1] = 'E-' + attr
                            
                        elif length > 2 and char_ner_tags[j:k] == ['O']*length:
                            char_ner_tags[j] = 'B-' + attr
                            char_ner_tags[k-1] = 'E-' + attr
                            char_ner_tags[j+1:k-1] = ['I-' + attr]*(length-2)
                            
                        j = k
                    else:
                        j += 1
                

            q_words = zip(word_tokens, word_pos_tags, word_ner_tags)
            q_chars = zip(char_tokens, char_pos_tags, char_ner_tags)

            bioe_words.append(list(q_words))
            bioe_chars.append(list(q_chars))
            
        return bioe_words, bioe_chars