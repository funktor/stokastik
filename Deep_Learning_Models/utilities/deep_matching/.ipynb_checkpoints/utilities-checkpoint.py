import os, random, numpy as np, math
import constants.deep_matching.constants as cnt
import shared_utilities as shutils
from gensim.models import Word2Vec, FastText

def get_all_tokens_for_vector(items, char_tokens=False):
    wm_text = [str(item[3]) for item in items]
    cm_text = [str(item[5]) for item in items]
    
    all_titles = wm_text + cm_text
    all_tokens = [shutils.padd_fn(shutils.get_tokens(title, char_tokens=char_tokens), max_len=cnt.MAX_WORDS) for title in all_titles]
    
    return all_tokens

def get_tokens_indices(items, indices):
    data_pairs = []
    for i in indices:
        wm_title = str(items[i][3])
        cm_title = str(items[i][5])
        
#         wm_desc = str(items[i][4])
#         cm_desc = str(items[i][6])

        wm_word_tokens = shutils.padd_fn(shutils.get_tokens(wm_title, char_tokens=False), max_len=cnt.MAX_WORDS)
        cm_word_tokens = shutils.padd_fn(shutils.get_tokens(cm_title, char_tokens=False), max_len=cnt.MAX_WORDS)
        
        wm_char_tokens = [shutils.padd_fn(shutils.get_tokens(token, char_tokens=True), max_len=cnt.MAX_CHARS) for token in wm_word_tokens]
        cm_char_tokens = [shutils.padd_fn(shutils.get_tokens(token, char_tokens=True), max_len=cnt.MAX_CHARS) for token in cm_word_tokens]

        label = int(items[i][-1])

        data_pairs.append((wm_word_tokens, cm_word_tokens, wm_char_tokens, cm_char_tokens, label))
        
    return data_pairs

def get_vector_model(vector_model_id='wv', char_tokens=False):
    if vector_model_id == 'fasttext':
        if char_tokens:
            return FastText.load(cnt.FAST_TEXT_PATH_CHAR)
        return FastText.load(cnt.FAST_TEXT_PATH_WORD)
    else:
        if char_tokens:
            return Word2Vec.load(cnt.WORD_VECT_PATH_CHAR)
        return Word2Vec.load(cnt.WORD_VECT_PATH_WORD)
    