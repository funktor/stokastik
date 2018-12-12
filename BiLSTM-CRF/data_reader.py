import json, pickle, os
from collections import defaultdict
import numpy as np, random, re, math, pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
import tagger as tg

def clean_tokens(tokens, to_replace='[^\w\-\+\&\.\'\"]+'):
    lemma = WordNetLemmatizer()
    
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    tokens = [lemma.lemmatize(token) for token in tokens]
    
    return tokens

def tokenize(mystr):
    tokenizer = RegexpTokenizer('[^ ]+')
    return tokenizer.tokenize(mystr)

def get_tokens(sentence, to_replace='[^\w\-\+\&\.\'\"]+'):
    sentence = re.sub('<[^<]+?>', ' ', sentence)
    sentence = re.sub(to_replace, ' ', sentence)
    
    tokens = clean_tokens(tokenize(sentence), to_replace)
    tokens = [token.strip() for token in tokens]
    
    return tokens

def get_unique_items_pt(product_data, pt):
    items = [x for x in product_data if x[1] == pt]
    
    unique_items, added_item_ids = [], set()

    for item in items:
        if item[0] not in added_item_ids:
            unique_items.append(item)
            added_item_ids.add(item[0])
    
    return unique_items

def get_text_data(product_data_items):
    titles = [re.sub('<[^<]+?>', ' ', str(x[2])) for x in product_data_items]
    return titles

def get_item_text_from_input_file(input_file):
    product_data = []

    for df_chunk in pd.read_csv(input_file, sep='\\t', chunksize=10**4, engine='python', encoding='utf-8'):
        jsons = list(df_chunk[u'product_json'])

        for item in jsons:
            item = json.loads(item)

            attributes = item['product_attributes']

            attr_val_pairs = dict()

            for attr_key, vals in attributes.items():
                value = vals['values'][0]['value']
                attr_val_pairs[attr_key] = value

            item_id = attributes['item_id']['values'][0]['value'] if 'item_id' in attributes else None
            title = attributes['product_name']['values'][0]['value'] if 'product_name' in attributes else None
            long_desc = attributes['product_long_description']['values'][0]['value'] if 'product_long_description' in attributes else None
            shrt_desc = attributes['product_short_description']['values'][0]['value'] if 'product_short_description' in attributes else None
            pt = attributes['product_type']['values'][0]['value'] if 'product_type' in attributes else None

            product_data.append((item_id, pt, title, long_desc, shrt_desc, attr_val_pairs))
    
    items = get_unique_items_pt(product_data, "Laptop Computers")
    item_text = get_text_data(items)
    
    return items, item_text

def get_sequence_labels(text_chunks, phrase_chunks, labels, tag):
    n = len(phrase_chunks)
    
    if n > 0:
        b_tag, i_itag, e_tag = 'B-'+tag, 'I-'+tag, 'E-'+tag
        s_tag = 'S-'+tag

        for start in range(len(text_chunks)-n+1):
            a = [x.lower() for x in text_chunks[start:start+n]]
            b = [x.lower() for x in phrase_chunks]
            
            if a == b:
                if n == 1:
                    labels[start] = s_tag
                else:
                    labels[start+1:start+n-1] = [i_itag]*(n-2)
                    labels[start] = b_tag
                    labels[start+n-1] = e_tag
    
    return labels

def get_data(items, item_text, num_tokens=50):
    sentences, labels = [], []

    for idx in range(len(item_text)):
        item_txt = item_text[idx]
        text_chunks = get_tokens(item_txt)

        attr_vals = items[idx][5]

        label = ['O' for x in text_chunks]

        brand = tg.tag_brand_name(item_txt, attr_vals)
        if brand is not None:
            label = get_sequence_labels(text_chunks, get_tokens(brand), label, 'brand')

        screen = tg.tag_screen_size(item_txt, attr_vals)
        if screen is not None:
            label = get_sequence_labels(text_chunks, get_tokens(screen), label, 'screen')

        hdd = tg.tag_hdd_capacity(item_txt, attr_vals)
        if hdd is not None:
            label = get_sequence_labels(text_chunks, get_tokens(hdd), label, 'hdd')

        ram = tg.tag_ram_size(item_txt, attr_vals)
        if ram is not None:
            label = get_sequence_labels(text_chunks, get_tokens(ram), label, 'ram')

        res = tg.tag_screen_resolution(item_txt, attr_vals)
        if res is not None:
            label = get_sequence_labels(text_chunks, get_tokens(res), label, 'res')

        proc_s = tg.tag_processor_speed(item_txt, attr_vals)
        if proc_s is not None:
            label = get_sequence_labels(text_chunks, get_tokens(proc_s), label, 'proc_s')

        proc = tg.tag_processor_type(item_txt, attr_vals)
        if proc is not None:
            label = get_sequence_labels(text_chunks, get_tokens(proc), label, 'proc')
            
        if len(label) < num_tokens:
            n = num_tokens - len(label)
            labels.append(label + ['O']*n)
            sentences.append(text_chunks + ['PAD_TXT']*n)
        else:
            labels.append(label[:num_tokens])
            sentences.append(text_chunks[:num_tokens])
    
    return np.asarray(sentences), np.asarray(labels)

def get_train_test_data_reader_obj():
    data_file_path = 'data/train_test_data.pkl'

    if os.path.exists(data_file_path) is False:
        dr = DataReader()
        dr.read_data('data/pcf_dump_laptops.tsv')

        print "Saving datareader object..."
        with open(data_file_path, 'wb') as f:
            pickle.dump(dr, f)
    else:
        print "Loading datareader object..."
        with open(data_file_path, 'rb') as f:
            dr = pickle.load(f)
            
    return dr

class DataReader(object):
    def __init__(self):
        self.train_sents, self.test_sents = None, None
        self.train_labels, self.test_labels = None, None
        self.vocab, self.tags = None, None
        self.word2idx, self.tag2idx, self.idx2tag = None, None, None
        self.sent_seq_tr, self.sent_seq_te, self.tag_seq_tr, self.tag_seq_te = None, None, None, None
        self.max_words = None
        
    def read_data(self, input_file_path, test_pct=0.2, num_tokens=50):
        self.max_words = num_tokens
        
        print "Reading from input file path..."
        items, item_text = get_item_text_from_input_file(input_file_path)
        
        print "Getting sentences and labels..."
        sentences, labels = get_data(items, item_text, num_tokens=num_tokens)

        print "Splitting data..."
        train_indices, test_indices = train_test_split(range(len(labels)), test_size=test_pct)

        self.train_sents, self.test_sents = sentences[train_indices], sentences[test_indices]
        self.train_labels, self.test_labels = labels[train_indices], labels[test_indices]
        
        print "Getting vocabulary and tags..."
        vocab = set()
        for sent in sentences:
            vocab.update([token for token in sent])
            
        self.vocab = sorted(list(vocab))
        
        tags = set()
        for label in labels:
            tags.update([token for token in label])

        self.tags = sorted(list(tags))
        
        print "Generating inverted indices for vocab and tags..."
        self.word2idx = {w: i + 1 for i, w in enumerate(self.vocab)}
        self.tag2idx = {w: i for i, w in enumerate(self.tags)}
        self.idx2tag = {i: w for i, w in enumerate(self.tags)}
        
        print "Preprocessing for LSTM..."
        sent_seq = [[self.word2idx[w] for w in s] for s in sentences]
        sent_seq = pad_sequences(maxlen=num_tokens, sequences=sent_seq, padding="post", value=len(self.vocab)-1)
        
        tag_seq = [[self.tag2idx[w] for w in label] for label in labels]
        tag_seq = pad_sequences(maxlen=num_tokens, sequences=tag_seq, padding="post", value=self.tag2idx["O"])
        tag_seq = [to_categorical(i, num_classes=len(self.tags)) for i in tag_seq]
        
        sent_seq, tag_seq = np.asarray(sent_seq), np.asarray(tag_seq)
        
        self.sent_seq_tr, self.sent_seq_te, self.tag_seq_tr, self.tag_seq_te = sent_seq[train_indices], sent_seq[test_indices], tag_seq[train_indices], tag_seq[test_indices]
