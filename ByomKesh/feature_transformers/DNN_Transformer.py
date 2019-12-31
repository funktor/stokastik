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
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def get_preprocessed_data(sentences, feature_set=None, tokenizer=None, max_length=200):
    p_sents = []
    for sent in sentences:
        tokens = utils.get_tokens(sent, min_ngram=1, max_ngram=1)
        if feature_set is not None:
            tokens = [token.strip() for token in tokens if token in feature_set]
        p_sents += [' '.join(tokens)]
    
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(p_sents)

    tensor = tokenizer.texts_to_sequences(p_sents)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length, padding='post')
    
    return tensor, tokenizer


def create_pairs(class_labels, max_pairs=10000):
    data_pairs = zip(range(len(class_labels)), class_labels)
    
#     class_labels_dict = {}
    
#     for i in range(len(class_labels)):
#         m_labels = class_labels[i]
        
#         for label in m_labels:
#             if label not in class_labels_dict:
#                 class_labels_dict[label] = []
#             class_labels_dict[label].append(i)
    
#     data_pairs, all_data = [], set(range(len(class_labels)))
    
#     for cl, pos_i in class_labels_dict.items():
#         neg_i = list(all_data-set(pos_i))
        
#         for i in range(len(pos_i)-1):
#             u = random.sample(pos_i, min(len(pos_i), 100))
#             for j in u:
#                 data_pairs += [(pos_i[i], j, 1)]
        
#         for i in range(len(pos_i)):
#             u = random.sample(neg_i, min(len(neg_i), 100))
#             for j in u:
#                 data_pairs += [(pos_i[i], j, 0)]
    
    return np.array(data_pairs)


def data_generator(data_pairs, tensor, batch_size=64):
    num_batches = int(math.ceil(len(data_pairs)/batch_size))
    tensor = np.array(tensor)
    
    np.random.shuffle(data_pairs)
    batch_num = 0
    
    while True:
        m = batch_num % num_batches
        
        start, end = batch_size*m, min(len(data_pairs), batch_size*(m+1))
#         i, j, labels = zip(*data_pairs[start:end])
        i, labels = zip(*data_pairs[start:end])
        
        items_data = tensor[list(i)]
        
        batch_num += 1
        
        yield np.array(items_data), np.array(labels)
        

def get_shared_model(max_words, vocab_size, embed_dim):
    input = Input(shape=(max_words,)) #None, 200
    nlayer = Embedding(vocab_size, embed_dim, input_length=max_words)(input) #None, 200, 128
    
    nlayer = Conv1D(64, 11, activation='relu', padding='same')(nlayer) #None, 200, 32
    nlayer = Conv1D(64, 11, activation='relu', padding='same')(nlayer) #None, 200, 32
    nlayer = BatchNormalization()(nlayer)
    nlayer = MaxPooling1D(2)(nlayer) #None, 100, 32
    
    nlayer = Conv1D(128, 7, activation='relu', padding='same')(nlayer) #None, 100, 64
    nlayer = Conv1D(128, 7, activation='relu', padding='same')(nlayer) #None, 100, 64
    nlayer = BatchNormalization()(nlayer)
    nlayer = MaxPooling1D(2)(nlayer) #None, 50, 64
    
    nlayer = Conv1D(256, 5, activation='relu', padding='same')(nlayer) #None, 50, 64
    nlayer = Conv1D(256, 5, activation='relu', padding='same')(nlayer) #None, 50, 64
    nlayer = BatchNormalization()(nlayer)
    nlayer = MaxPooling1D(2)(nlayer) #None, 50, 64
    
    nlayer = Conv1D(512, 3, activation='relu', padding='same')(nlayer) #None, 50, 128
    nlayer = Conv1D(512, 3, activation='relu', padding='same')(nlayer) #None, 50, 128
    nlayer = BatchNormalization()(nlayer)
    output = GlobalAveragePooling1D()(nlayer) #None, 1, 128
    
    model = Model(input, output)
    return model


def get_dnn_embeddings(sentences, model, feature_set, dnn_tokenizer, max_length):
    tensors, _ = get_preprocessed_data(sentences, feature_set=feature_set, tokenizer=dnn_tokenizer, max_length=max_length)
    tensors = np.array(tensors)
    return model.transform(tensors)


class DNNFeatures():
    def __init__(self, model_path, embed_dim=128, max_words=200, batch_size=64, num_epochs=64, num_classes=None, features=None):
        self.model_path = model_path
        self.model = None
        self.vocab_size = None
        self.embed_dim = embed_dim
        self.max_words = max_words
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.features = features
        self.dnn_tokenizer = None
        self.num_classes = num_classes
        self.classes = None
    
    def init_model(self):
        input = Input(shape=(self.max_words,))
        
        shared_model = get_shared_model(self.max_words, self.vocab_size, self.embed_dim)
        
        nlayer = shared_model(input)
        nlayer = BatchNormalization()(nlayer)
        out = Dense(self.num_classes, activation="softmax")(nlayer)

        self.model = Model(input, out)
        adam = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    def fit(self, X, y=None):
        input_tensor, self.dnn_tokenizer = get_preprocessed_data(np.asarray(X), feature_set=set(self.features), max_length=self.max_words)
        input_tensor = np.array(input_tensor)
        
        label_transformer = LabelBinarizer()
        transformed_labels = label_transformer.fit_transform(y)
        self.classes = np.array(label_transformer.classes_)

        X_train, X_valid, y_train, y_valid = train_test_split(input_tensor, transformed_labels, test_size=0.2)
        
        self.vocab_size = len(self.dnn_tokenizer.word_index)+1
        
        self.init_model()

        callbacks = [
            ModelCheckpoint(filepath=self.model_path, monitor='loss', save_best_only=True),
        ]
        
        self.model.fit(X_train, y_train, 
                       batch_size=self.batch_size, 
                       epochs=self.num_epochs, verbose=1, 
                       callbacks=callbacks, 
                       validation_data=(X_valid, y_valid),
                       shuffle=True)
        
        preds = self.model.predict(X_valid)
        pred_labels = self.classes[np.argmax(preds, axis=1)]
        
        true_labels = self.classes[np.argmax(y_valid, axis=1)]
        
        print(classification_report(true_labels, pred_labels))
        
    def predict(self, X):
        X, _ = get_preprocessed_data(np.asarray(X), feature_set=set(self.features), tokenizer=self.dnn_tokenizer, max_length=self.max_words)
        X = np.array(X)
        
        self.vocab_size = len(self.dnn_tokenizer.word_index)+1
        
        self.init_model()
        self.model.load_weights(self.model_path)
        
        preds = self.model.predict(X)
        return self.classes[np.argmax(preds, axis=1)]
        
    
    def transform(self, X):
        X, _ = get_preprocessed_data(np.asarray(X), feature_set=set(self.features), tokenizer=self.dnn_tokenizer, max_length=self.max_words)
        X = np.array(X)
        
        self.vocab_size = len(self.dnn_tokenizer.word_index)+1
        
        self.init_model()
        self.model.load_weights(self.model_path)
        
        embeddings = K.function([self.model.layers[0].input, self.model.layers[1].input, self.model.layers[2].layers[0].input], [self.model.layers[3].get_output_at(0)])
        return embeddings([X, X, X])[0]
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)