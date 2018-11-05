import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

import pickle, os, utils
from lstm_models import BiLSTM
import data_reader

dr = data_reader.get_train_test_data_reader_obj()
        
lstm_model_path = 'data/lstm.h5'
model = BiLSTM(vocab_size=len(dr.vocab), max_words=dr.max_words, num_tags=len(dr.tags), embedding_size=64, model_file_path=lstm_model_path)
model.build_model()
model.fit(dr.sent_seq_tr, dr.tag_seq_tr)
model.save()

print model.score(dr.sent_seq_te, dr.tag_seq_te, dr.idx2tag)
