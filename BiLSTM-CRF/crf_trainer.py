from collections import defaultdict
import numpy as np, random, re, math, pickle
import pycrfsuite

def get_word_features(word, word_id):
    word_lower = word.lower()
    
    return [word_id + '=' + word, 
            word_id + '.lower=%s' % word_lower, 
            word_id + '.is_digit=%s' % word.isdigit(), 
            word_id + '.has_digit=%s' % (bool(re.search('\d+', word))), 
            word_id + '.is_number=%s' % (bool(re.match('\d+\.\d+|\d+', word))), 
            word_id + '.is_title=%s' % word.istitle(), 
            word_id + '.is_first_capital=%s' % (bool(re.match('^[A-Z].*$', word))), 
            word_id + '.is_gb_tb=%s' % (bool(re.search('(\d+\.\d+|\d+)\s*gb|tb', word_lower))), 
            word_id + '.is_in=%s' % (bool(re.match('(\d+\.\d+|\d+)\"', word_lower))), 
            word_id + '.is_ghz=%s' % (bool(re.match('(\d+\.\d+|\d+)[- ]*ghz', word_lower))), 
            word_id + '.is_proc_pat=%s' % (bool(re.search('([A-Za-z][0-9]+[- ][0-9]{4}[A-Za-z]+)', word_lower)))]

def word2features(sentence, pos):
    features = get_word_features(sentence[pos], 'curr_word')
    
    if pos > 0:
        features += get_word_features(sentence[pos - 1], 'prev_word')
    else:
        features.append('BOS')
        
    if pos < len(sentence) - 1:
        features += get_word_features(sentence[pos + 1], 'next_word')
    else:
        features.append('EOS')
        
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def train_crf_model(train_sentences, train_labels, model_file='laptop.crfsuite'):
    print "Loading CRF Trainer..."
    train_features = [sent2features(sent) for sent in train_sentences]
    
    model = pycrfsuite.Trainer(verbose=True)

    for xseq, yseq in zip(train_features, train_labels):
        model.append(xseq, yseq)
        
    model.set_params({
        'c1': 1.0,
        'c2': 0.001,
        'max_iterations': 200,
        'feature.possible_transitions': True
    })
    
    print "Training CRF..."
    model.train(model_file)
    
    return model