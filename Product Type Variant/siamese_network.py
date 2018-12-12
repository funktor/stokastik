import keras
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, Bidirectional, InputSpec, Lambda, Average
from keras.models import load_model
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from sklearn.metrics import classification_report
import random
from sklearn.cross_validation import train_test_split
import common_utils as utils

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[2], 1), initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros', trainable=True)
        
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        scores = K.dot(x, self.kernel) + self.bias
        #output_shape = (N, 200, 1)
        
        weights = K.softmax(scores, axis=1)
        #output_shape = (N, 200, 1)
        
        weighted_avg = keras.layers.dot([x, weights], axes=1)
        #output_shape = (N, 128, 1)
        
        return weighted_avg[:,:,0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
        
class SiameseNet(object):
    def __init__(self, vocab_size, max_words, embedding_size, model_file_path):
        self.vocab_size = vocab_size
        self.max_words = max_words
        self.embedding_size = embedding_size
        self.model = None
        self.model_file_path = model_file_path
        
    def build_model(self):
        print "Building model..."
        
        input1 = Input(shape=(self.max_words,))
        input2 = Input(shape=(self.max_words,))
        
        embed_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_size, input_length=self.max_words, mask_zero=True)
        
        embed1 = embed_layer(input1)
        embed2 = embed_layer(input2)
        
        #output_shape = (N, 200, 64)
        
        bilstm_layer = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))
        
        bilstm_w1 = bilstm_layer(embed1)
        bilstm_w2 = bilstm_layer(embed2)
        
        #output_shape = (N, 200, 128)
        
        attn_layer = AttentionLayer()
        
        attention_w1 = attn_layer(bilstm_w1)
        attention_w2 = attn_layer(bilstm_w2)
        
        #output_shape = (N, 128)
        
        out = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))([attention_w1, attention_w2])
        
        #output_shape = (N, 1)

        self.model = Model([input1, input2], out)
        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        
    def fit(self, X1, X2, y):
        print "Fitting model..."
        self.model.fit([X1, X2], np.array(y), batch_size=32, epochs=10, validation_split=0.1, verbose=1)
        
    def save(self):
        self.model.save(self.model_file_path)
        
    def load(self):
        self.model.load_weights(self.model_file_path)
        
    def predict(self, X1, X2):
        print "Predicting..."
        return self.model.predict([X1, X2], verbose=1)
    
    def get_embeddings(self, X):
        print "Predicting..."
        embeddings = K.function([self.model.layers[0].input, self.model.layers[1].input], [self.model.layers[4].get_output_at(0)])
                                
        return embeddings([X, X])[0]
    
    def score(self, X1, X2, y):
        test_pred = np.rint(self.predict(X1, X2)).astype(int)
        return classification_report(y, test_pred)