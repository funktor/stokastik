from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.models import load_model
import utils
import numpy as np

def pred2label(pred, tag_inverse_transformer):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(tag_inverse_transformer[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

class BiLSTM(object):
    def __init__(self, vocab_size, max_words, num_tags, embedding_size, model_file_path):
        self.vocab_size = vocab_size
        self.max_words = max_words
        self.num_tags = num_tags
        self.embedding_size = embedding_size
        self.model = None
        self.model_file_path = model_file_path
        
    def build_model(self):
        print "Building model..."
        input = Input(shape=(self.max_words,))
        embed = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_size, input_length=self.max_words, mask_zero=True)(input)
        bilstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(embed)
        dense = TimeDistributed(Dense(50, activation="relu"))(bilstm)
        out = TimeDistributed(Dense(self.num_tags, activation="softmax"))(dense)

        self.model = Model(input, out)
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        
    def fit(self, X, y):
        print "Fitting model..."
        self.model.fit(X, np.array(y), batch_size=32, epochs=15, validation_split=0.1, verbose=1)
        
    def save(self):
        self.model.save(self.model_file_path)
        
    def load(self):
        self.model.load_weights(self.model_file_path)
        
    def predict(self, X):
        print "Predicting..."
        return self.model.predict(X, verbose=1)
    
    def score(self, X, y, tag_inverse_transformer):
        test_pred = self.predict(X)
        pred_labels, test_labels = pred2label(test_pred, tag_inverse_transformer), pred2label(y, tag_inverse_transformer)
        
        print utils.get_accuracy(test_labels, pred_labels)
        print
        
        return utils.get_classification_score(test_labels, pred_labels)
    
    
class BiLSTM_CRF(BiLSTM):
    def __init__(self, vocab_size, max_words, num_tags, embedding_size, model_file_path):
        super(BiLSTM_CRF, self).__init__(vocab_size, max_words, num_tags, embedding_size, model_file_path)
        
    def build_model(self):
        print "Building model..."
        input = Input(shape=(self.max_words,))
        embed = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_size, input_length=self.max_words, mask_zero=True)(input)
        bilstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(embed)
        dense = TimeDistributed(Dense(50, activation="relu"))(bilstm)
        crf = CRF(self.num_tags)
        out = crf(dense)

        self.model = Model(input, out)
        self.model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
