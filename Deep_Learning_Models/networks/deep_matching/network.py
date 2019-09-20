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
from sklearn.metrics import classification_report
import constants.deep_matching.constants as cnt
import shared_utilities as shutils
import utilities.deep_matching.utilities as utils
from keras_self_attention import SeqSelfAttention

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
    
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[2], input_shape[2]), initializer='uniform', trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        a = K.dot(x, self.kernel)
        cross_weights = K.batch_dot(x, K.permute_dimensions(a, (0,2,1)))
        cross_weights = cross_weights/K.sum(cross_weights, axis=None)
        
        p = K.batch_dot(cross_weights, x)
        return K.tanh(concatenate([p, x]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]*2)
    
    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
        
        
class CustomMerge(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(CustomMerge, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CustomMerge, self).build(input_shape)

    def call(self, x):
        return dot([x[0], x[1]], axes=-1)
        
#         _, u, v = K.int_shape(cross_weights)
#         p = Reshape((u*v,))(cross_weights)
        
#         a = K.repeat_elements(x[0], x[0].shape[1], axis=1)
        
#         b = K.tile(x[1], [1, x[1].shape[1], 1])
        
#         z = concatenate([a, b], axis=-1)
        
#         return dot([p, z], axes=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][1])
    
    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


def get_shared_model():
    input_word = Input(shape=(cnt.MAX_WORDS, cnt.WORD_VECTOR_DIM))
    input_char = Input(shape=(cnt.MAX_WORDS, cnt.MAX_CHARS, cnt.CHAR_VECTOR_DIM))
    
    input_char_reshape = Reshape((cnt.MAX_WORDS*cnt.MAX_CHARS, cnt.CHAR_VECTOR_DIM))(input_char)
    
    convd_char = Conv1D(32, 3, padding='same', activation='relu')(input_char_reshape) #None, cnt.MAX_WORDS*cnt.MAX_CHARS, 32
    unflatten_char = Reshape((cnt.MAX_WORDS, cnt.MAX_CHARS, 32))(convd_char) #None, cnt.MAX_WORDS, cnt.MAX_CHARS, 32
    
    convd_char_max = TimeDistributed(GlobalMaxPool1D())(unflatten_char) #None, cnt.MAX_WORDS, 1, 32
    convd_char_max = Reshape((cnt.MAX_WORDS, 32))(convd_char_max) #None, cnt.MAX_WORDS, 32
    
    concatenated = concatenate([convd_char_max, input_word], axis=-1) #None, cnt.MAX_WORDS, 128+32=160
    
    nlayer = Conv1D(32, 3, activation='relu', padding='same')(concatenated) #None, cnt.MAX_WORDS, 32
    nlayer = Conv1D(32, 3, activation='relu', padding='same')(nlayer) #None, cnt.MAX_WORDS, 32
    nlayer = BatchNormalization()(nlayer)
    output = MaxPooling1D(2)(nlayer) #None, cnt.MAX_WORDS/2, 32
    
    model = Model([input_word, input_char], output)
    return model


class DeepMatchingNetwork:
    def __init__(self, data_generator=None, num_train=None, num_test=None):
        self.data_generator = data_generator
        self.num_test = num_test
        self.num_train = num_train
        self.model = None
    
    def init_model(self):
        input_word_1 = Input(shape=(cnt.MAX_WORDS, cnt.WORD_VECTOR_DIM))
        input_word_2 = Input(shape=(cnt.MAX_WORDS, cnt.WORD_VECTOR_DIM))
        
        input_char_1 = Input(shape=(cnt.MAX_WORDS, cnt.MAX_CHARS, cnt.CHAR_VECTOR_DIM))
        input_char_2 = Input(shape=(cnt.MAX_WORDS, cnt.MAX_CHARS, cnt.CHAR_VECTOR_DIM))
        
        shared_model = get_shared_model()

        nlayer1 = shared_model([input_word_1, input_char_1]) #None, 50, 32
        nlayer2 = shared_model([input_word_2, input_char_2]) #None, 50, 32
        
        merge = dot([nlayer1, nlayer2], axes=-1) #None, 50, 50
        
        shp = K.int_shape(merge)
        merge = Reshape((shp[1], shp[2], 1))(merge) #None, 50, 50, 1
        
        nlayer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(merge) #None, 50, 50, 64
        nlayer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(nlayer) #None, 50, 50, 64
        nlayer = BatchNormalization()(nlayer)
        nlayer = GlobalAveragePooling2D()(nlayer) #None, 1, 1, 64

        out = Dense(1, activation="sigmoid")(nlayer)

        self.model = Model([input_word_1, input_word_2, input_char_1, input_char_2], out)
        adam = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    def fit(self):
        self.init_model()

        callbacks = [
            ModelCheckpoint(filepath=cnt.MODEL_PATH, monitor='val_loss', save_best_only=True),
        ]

        self.model.fit_generator(self.data_generator(self.num_train, 'train'),
                                 callbacks=callbacks, 
                                 steps_per_epoch=shutils.get_num_batches(self.num_train, cnt.BATCH_SIZE), 
                                 validation_data=self.data_generator(self.num_test, 'test'), 
                                 validation_steps=shutils.get_num_batches(self.num_test, cnt.BATCH_SIZE), 
                                 epochs=cnt.NUM_EPOCHS, verbose=1, use_multiprocessing=True)
    
    def predict(self, test_data, return_probability=False):
        preds = self.predict_probability(test_data)
        preds = [x[0] for x in preds]
        if return_probability:
            return np.rint(preds).astype(int), preds
        
        return np.rint(preds).astype(int)
    
    def predict_probability(self, test_data):
        return self.model.predict(test_data)
    
    def scoring(self):
        self.init_model()
        self.model.load_weights(cnt.MODEL_PATH)
        
        test_labels, pred_labels, total_batches = [], [], shutils.get_num_batches(self.num_test, cnt.BATCH_SIZE)
        
        num_batches = 0
        for batch_data, batch_labels in self.data_generator(self.num_test, 'test'):
            test_labels += batch_labels.tolist()
            pred_labels += self.predict(batch_data).tolist()
            num_batches += 1
            if num_batches == total_batches:
                break
        
        print(classification_report(test_labels, pred_labels))
        
    def save(self):
        self.model.save(cnt.MODEL_PATH)
        
    def load(self):
        self.model.load_weights(cnt.MODEL_PATH)
