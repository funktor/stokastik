import tensorflow as tf
import keras, os
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, Bidirectional, InputSpec, Lambda, Average, CuDNNLSTM, Flatten, TimeDistributed, Dropout, concatenate, dot, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, UpSampling2D, UpSampling1D, AveragePooling1D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPool1D
from keras.models import load_model
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import pickle, os, re, numpy as np, gensim, time, sys
import pandas as pd, math, collections, random, tables
import Constants as cnt
import Utilities as utils

def get_preprocessed_data(sentences, feature_set=None, tokenizer=None, max_length=200):
    p_sents = []
    for sent in sentences:
        tokens = utils.get_tokens(sent, min_ngram=1, max_ngram=1)
        if feature_set is not None:
            tokens = [token for token in tokens if token in feature_set]
        p_sents += [' '.join(tokens)]
    
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(p_sents)

    tensor = tokenizer.texts_to_sequences(p_sents)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length, padding='post')
    
    return tensor, tokenizer


def create_pairs(class_labels, max_pairs=10000):
    class_labels_dict = {}
    
    for i in range(len(class_labels)):
        m_labels = class_labels[i]
        
        for label in m_labels:
            if label not in class_labels_dict:
                class_labels_dict[label] = []
            class_labels_dict[label].append(i)
    
    data_pairs, all_data = [], set(range(len(class_labels)))
    
    for cl, pos_i in class_labels_dict.items():
        neg_i = list(all_data-set(pos_i))
        
        for i in range(len(pos_i)-1):
            u = random.sample(pos_i, min(len(pos_i), 500))
            for j in u:
                data_pairs += [(pos_i[i], j, 0)]
        
        for i in range(len(pos_i)):
            u = random.sample(neg_i, min(len(neg_i), 500))
            for j in u:
                data_pairs += [(pos_i[i], j, 1)]
    
    return np.array(data_pairs)


def data_generator(data_pairs, tensor, batch_size=64):
    num_batches = int(math.ceil(len(data_pairs)/batch_size))
    tensor = np.array(tensor)
    
    np.random.shuffle(data_pairs)
    batch_num = 0
    
    while True:
        m = batch_num % num_batches
        
        start, end = batch_size*m, min(len(data_pairs), batch_size*(m+1))
        i, j, labels = zip(*data_pairs[start:end])
        
        items_data_1 = tensor[list(i)]
        items_data_2 = tensor[list(j)]
        
        batch_num += 1
        
        yield [np.array(items_data_1), np.array(items_data_2)], np.array(labels)
        

def get_shared_model(max_words, vocab_size, embed_dim):
    input = Input(shape=(max_words,)) #None, 200
    nlayer = Embedding(vocab_size, embed_dim, input_length=max_words)(input) #None, 200, 128
    
    nlayer = Conv1D(32, 3, activation='relu', padding='same')(nlayer) #None, 200, 32
    nlayer = Conv1D(32, 3, activation='relu', padding='same')(nlayer) #None, 200, 32
    nlayer = BatchNormalization()(nlayer)
    nlayer = MaxPooling1D(2)(nlayer) #None, 100, 32
    
    nlayer = Conv1D(64, 3, activation='relu', padding='same')(nlayer) #None, 100, 64
    nlayer = Conv1D(64, 3, activation='relu', padding='same')(nlayer) #None, 100, 64
    nlayer = BatchNormalization()(nlayer)
    nlayer = MaxPooling1D(2)(nlayer) #None, 50, 64
    
    nlayer = Conv1D(128, 3, activation='relu', padding='same')(nlayer) #None, 50, 128
    nlayer = Conv1D(128, 3, activation='relu', padding='same')(nlayer) #None, 50, 128
    nlayer = BatchNormalization()(nlayer)
    output = GlobalAveragePooling1D()(nlayer) #None, 1, 128
    
    model = Model(input, output)
    return model


def get_dnn_embeddings(sentences, model, feature_set, dnn_tokenizer, max_length):
    tensors, _ = get_preprocessed_data(sentences, feature_set=feature_set, tokenizer=dnn_tokenizer, max_length=max_length)
    tensors = np.array(tensors)
    return model.transform(tensors)


class DNN():
    def __init__(self, model_path, data_generator=None, num_train=None, vocab_size=None, embed_dim=128, 
                 max_words=200, data_pairs=None, input_tensor=None, batch_size=64, num_epochs=64):
        
        self.model_path = model_path
        self.data_generator = data_generator
        self.num_train = num_train
        self.model = None
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_words = max_words
        self.data_pairs = data_pairs
        self.input_tensor = input_tensor
        self.batch_size = batch_size
        self.num_batches = int(math.ceil(self.num_train/self.batch_size))
        self.num_epochs = num_epochs
    
    def init_model(self):
        input_sent_1 = Input(shape=(self.max_words,))
        input_sent_2 = Input(shape=(self.max_words,))
        
        shared_model = get_shared_model(self.max_words, self.vocab_size, self.embed_dim)
        
        nlayer1 = shared_model(input_sent_1)
        nlayer2 = shared_model(input_sent_2)
        
        nlayer = BatchNormalization()
        
        nlayer1 = nlayer(nlayer1)
        nlayer2 = nlayer(nlayer2)
        
        merge = dot([nlayer1, nlayer2], axes=1, normalize=True)
        
        nlayer = BatchNormalization()(merge)
        
        out = Dense(1, activation="linear")(nlayer)

        self.model = Model([input_sent_1, input_sent_2], out)
        adam = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    
    def fit(self):
        self.init_model()

        callbacks = [
            ModelCheckpoint(filepath=self.model_path, monitor='loss', save_best_only=True),
        ]
        
        print(self.num_batches)

        self.model.fit_generator(self.data_generator(self.data_pairs, self.input_tensor, self.batch_size),
                                 callbacks=callbacks, 
                                 steps_per_epoch=self.num_batches, 
                                 epochs=self.num_epochs, verbose=1)
    
    def transform(self, X):
        embeddings = K.function([self.model.layers[0].input, self.model.layers[1].input, self.model.layers[2].layers[0].input], [self.model.layers[3].get_output_at(0)])
        return embeddings([X, X, X])[0]