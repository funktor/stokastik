import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

import pickle, os, pycrfsuite, utils
import crf_trainer as crft
import data_reader

dr = data_reader.get_train_test_data_reader_obj()
        
crf_model_file='data/laptop.crfsuite'
crft.train_crf_model(dr.train_sents, dr.train_labels, model_file=crf_model_file)

tagger = pycrfsuite.Tagger()
tagger.open(crf_model_file)

test_features = [crft.sent2features(sent) for sent in dr.test_sents]
pred_labels = [tagger.tag(feats) for feats in test_features]

print utils.get_classification_score(dr.test_labels, pred_labels)
print
print utils.get_accuracy(dr.test_labels, pred_labels)
        
