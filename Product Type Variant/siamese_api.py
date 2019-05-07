from siamese_network import SiameseNet
import random, numpy as np
import pickle, tables, math, time, os
import grouping_utils as gutils
import constants as cnt
from gensim.models import Word2Vec

class SiameseAPI(object):
    def __init__(self):
        self.vocab_size = None
        self.model = None
        self.word2idx_map = None
        self.wv_model = Word2Vec.load(os.path.join(cnt.DATA_FOLDER, cnt.WV_MODEL_FILE))
        
    def get_model(self):
        self.word2idx_map = gutils.load_data_pkl(cnt.WORD2IDX_FILE)
        self.vocab_size = len(self.word2idx_map)
        
        num_train = len(gutils.load_data_pkl(cnt.TRAIN_DATA_PAIRS_FILE))
        num_test = len(gutils.load_data_pkl(cnt.TEST_DATA_PAIRS_FILE))
        num_validation = len(gutils.load_data_pkl(cnt.VALIDATION_DATA_PAIRS_FILE))
        
        siamese_net = SiameseNet(vocab_size=self.vocab_size, 
                                 training_samples=num_train, 
                                 validation_samples=num_validation, 
                                 testing_samples=num_test, 
                                 use_generator=True)
        self.model = siamese_net
        
    def train_model(self):
        self.get_model()
        self.model.fit()
        self.model.score()
    
    def get_representation(self, sentence):
        sent = gutils.padd_fn(gutils.get_tokens(sentence.encode("ascii", errors="ignore").decode()))
        return self.model.get_embeddings(gutils.get_wv_siamese(self.wv_model, [sent]))[0]
    
    def get_representations(self, sentences):
        sentences = [gutils.padd_fn(gutils.get_tokens(sentence.encode("ascii", errors="ignore").decode())) for sentence in sentences]
        return self.model.get_embeddings(gutils.get_wv_siamese(self.wv_model, sentences))
    
    def get_prediction(self, sentence1, sentence2):
        sent1 = gutils.padd_fn(gutils.get_tokens(sentence1.encode("ascii", errors="ignore").decode()))
        sent2 = gutils.padd_fn(gutils.get_tokens(sentence2.encode("ascii", errors="ignore").decode()))
        
        return self.model.predict([gutils.get_wv_siamese(self.wv_model, [sent1]), 
                                   gutils.get_wv_siamese(self.wv_model, [sent2])])[0][0]
    
    def get_distance_threshold(self, threshold=0.5):
        self.get_model()
        self.model.init_model()
        self.model.load()
        
        weight, bias = self.model.model.layers[9].get_weights()
        return -float(bias[0]+math.log((1.0/threshold)-1.0))/weight[0][0]
    
    def insert_embeddings_pytables(self):
        try:
            self.get_model()
            self.model.init_model()
            self.model.load()
            
            embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SIAMESE_EMBEDDINGS_FILE), mode='w')
            atom = tables.Float32Atom()
            embeds_arr = embeds_file.create_earray(embeds_file.root, 'data', atom, (0, cnt.SIAMESE_EMBEDDING_SIZE))
            
            sent_tokens_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SENT_TOKENS_FILE), mode='r')
            sent_tokens = sent_tokens_file.root.data
            
            n, batch_size = len(sent_tokens), cnt.PYTABLES_INSERT_BATCH_SIZE
            num_batches = int(math.ceil(float(n)/batch_size))
            
            for m in range(num_batches):
                start, end = m*batch_size, min((m+1)*batch_size, n)
                tokens_arr_input = gutils.get_wv_siamese(self.wv_model, sent_tokens[start:end,:])
                embeds = self.model.get_embeddings(tokens_arr_input)
                embeds_arr.append(embeds)
                
        finally:
            sent_tokens_file.close()
            embeds_file.close()
            
    def fetch_embeddings_pytables(self, item_indexes=None):
        try:
            embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SIAMESE_EMBEDDINGS_FILE), mode='r')
            embeds_arr = embeds_file.root.data
            
            if item_indexes is not None:
                output = np.array([embeds_arr[i] for i in item_indexes])
            else:
                output = np.array([embeds_arr[i] for i in range(len(embeds_arr))])
                
            return output
                
        finally:
            embeds_file.close()