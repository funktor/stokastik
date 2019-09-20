import keras, os
from keras.models import Model, Input
from keras.layers import Dense, Lambda, Flatten, Dropout, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.models import load_model
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import pickle, os, re, numpy as np, gensim, time, sys
import pandas as pd, math, collections, random, tables
from sklearn.metrics import classification_report, precision_recall_fscore_support
import constants.color_extraction.constants as cnt
import shared_utilities as shutils
import utilities.color_extraction.utilities as utils
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf

def img_model():
    input = Input(shape=(cnt.IMAGE_SIZE, cnt.IMAGE_SIZE, 3))
    n_layer = input
    
    n_layer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(n_layer) #None, 128, 128, 64
    n_layer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(n_layer) #None, 128, 128, 64
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer) #None, 64, 64, 64
    
    n_layer = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(n_layer) #None, 64, 64, 128
    n_layer = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(n_layer) #None, 64, 64, 128
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer) #None, 32, 32, 128
    
    n_layer = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(n_layer) #None, 32, 32, 128
    n_layer = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(n_layer) #None, 32, 32, 128
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer) #None, 16, 16, 128
    
    n_layer = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(n_layer) #None, 16, 16, 256
    n_layer = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(n_layer) #None, 16, 16, 256
    n_layer = BatchNormalization()(n_layer)
    
    n_layer = GlobalAveragePooling2D()(n_layer) #None, 1, 1, 256
    n_layer = BatchNormalization()(n_layer)
    
    model = Model(inputs=input, outputs=n_layer)
    return model


def txt_model():
    input = Input(shape=(cnt.MAX_WORDS, cnt.WORD_VECTOR_DIM))
    n_layer = input
    
    n_layer = Conv1D(32, 3, activation='relu', padding='same')(n_layer) #None, 70, 32
    n_layer = Conv1D(32, 3, activation='relu', padding='same')(n_layer) #None, 70, 32
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling1D(2)(n_layer) #None, 35, 32
    
    n_layer = Conv1D(64, 3, activation='relu', padding='same')(n_layer) #None, 35, 64
    n_layer = Conv1D(64, 3, activation='relu', padding='same')(n_layer) #None, 35, 64
    n_layer = BatchNormalization()(n_layer)
    n_layer = GlobalAveragePooling1D()(n_layer) #None, 1, 64
    
    model = Model(inputs=input, outputs=n_layer)
    return model


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        if gpus > 1:
            pmodel = multi_gpu_model(ser_model, gpus)
            self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


class ColorExtractionNetwork:
    def __init__(self, data_generator=None, num_train=None, num_test=None):
        self.data_generator = data_generator
        self.num_test = num_test
        self.num_train = num_train
        self.pt_model, self.color_model = None, None
        
    
    def init_pt_model(self):
        input_img = Input(shape=(cnt.IMAGE_SIZE, cnt.IMAGE_SIZE, 3))
        input_txt = Input(shape=(cnt.MAX_WORDS, cnt.WORD_VECTOR_DIM))
        
        n_layer1 = img_model()(input_img)
        n_layer2 = txt_model()(input_txt)
        
        n_layer = concatenate([n_layer1, n_layer2], axis=-1)
        
        out = Dense(cnt.NUM_PTS, activation="softmax")(n_layer)

        self.pt_model = Model(inputs=[input_img, input_txt], outputs=out)
        
        
    def init_color_model(self):
        input_img = Input(shape=(cnt.IMAGE_SIZE, cnt.IMAGE_SIZE, 3))
        input_txt = Input(shape=(cnt.MAX_WORDS, cnt.WORD_VECTOR_DIM))
        
        img = img_model()
        txt = txt_model()

        self.init_pt_model()
        self.pt_model.load_weights(cnt.PT_MODEL_PATH)

        img.set_weights(self.pt_model.layers[2].get_weights())
        txt.set_weights(self.pt_model.layers[3].get_weights())

        n_layer1 = img(input_img)
        n_layer2 = txt(input_txt)
        
        n_layer = concatenate([n_layer1, n_layer2], axis=-1)
        out = Dense(cnt.NUM_COLORS, activation="softmax")(n_layer)

        self.color_model = Model(inputs=[input_img, input_txt], outputs=out)
    
    
    def fit(self):
        with tf.device("/cpu:0"):
            self.init_pt_model()

        parallel_pt_model = ModelMGPU(self.pt_model, gpus=cnt.USE_NUM_GPUS)
        
        adam = optimizers.Adam(lr=0.001)
        parallel_pt_model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
    
        callbacks = [
            ModelCheckpoint(filepath=cnt.PT_MODEL_PATH, monitor='val_loss', save_best_only=True),
        ]

        parallel_pt_model.fit_generator(self.data_generator(self.num_train, 'train', 'pt'),
                                    callbacks=callbacks, 
                                    steps_per_epoch=shutils.get_num_batches(self.num_train, cnt.BATCH_SIZE), 
                                    validation_data=self.data_generator(self.num_test, 'test', 'pt'), 
                                    validation_steps=shutils.get_num_batches(self.num_test, cnt.BATCH_SIZE), 
                                    epochs=cnt.NUM_EPOCHS, verbose=1, use_multiprocessing=True)

        with tf.device("/cpu:0"):
            self.init_color_model()
            
        parallel_color_model = ModelMGPU(self.color_model, gpus=cnt.USE_NUM_GPUS)
        
        adam = optimizers.Adam(lr=0.001)
        parallel_color_model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

        callbacks = [
            ModelCheckpoint(filepath=cnt.COLOR_MODEL_PATH, monitor='val_loss', save_best_only=True),
        ]

        parallel_color_model.fit_generator(self.data_generator(self.num_train, 'train', 'color'),
                                       callbacks=callbacks, 
                                       steps_per_epoch=shutils.get_num_batches(self.num_train, cnt.BATCH_SIZE), 
                                       validation_data=self.data_generator(self.num_test, 'test', 'color'), 
                                       validation_steps=shutils.get_num_batches(self.num_test, cnt.BATCH_SIZE), 
                                       epochs=cnt.NUM_EPOCHS, verbose=1, use_multiprocessing=True)
        
    
    def predict(self, test_data, type='pt', return_probability=False):
        preds = self.predict_probability(test_data, type)
        outs, probs = [], []
        
        for i in range(preds.shape[0]):
            g, h = np.copy(preds[i]), np.copy(preds[i])
            g[h > 0.5] = 1
            g[h <= 0.5] = 0
            outs.append(g.tolist())
            probs.append(h[h > 0.5].tolist())
        
        outs = [[int(x) for x in y] for y in outs]
            
        if return_probability:
            return outs, probs
        
        return outs
    
    def predict_probability(self, test_data, type='pt'):
        if type == 'pt':
            return self.pt_model.predict(test_data)
        else:
            return self.color_model.predict(test_data)
    
    def scoring(self, type='pt', save_imgs=False, save_cams=False):
        test_labels, pred_labels, total_batches = [], [], shutils.get_num_batches(self.num_test, cnt.BATCH_SIZE)
        
        if type == 'pt':
            encoder = shutils.load_data_pkl(cnt.PT_ENCODER_PATH)
            pred_out_dir = cnt.PT_PREDS_PATH
            cam_dir = cnt.PT_CAMS_PATH
            self.init_pt_model()
            model = self.pt_model
            model.load_weights(cnt.PT_MODEL_PATH)
            
        else:
            encoder = shutils.load_data_pkl(cnt.COLOR_ENCODER_PATH)
            pred_out_dir = cnt.COLOR_PREDS_PATH
            cam_dir = cnt.COLOR_CAMS_PATH
            self.init_color_model()
            model = self.color_model
            model.load_weights(cnt.COLOR_MODEL_PATH)
        
        num_batches, start = 0, 0
        
        for batch_data, batch_labels in self.data_generator(self.num_test, 'test', type):
            test_labels += batch_labels.tolist()
            predictions = self.predict(batch_data, type)
            pred_labels += predictions
            num_batches += 1
            
            indices = [start + i for i in range(len(batch_labels))]
        
            if save_imgs:
                utils.save_imgs(batch_data, indices, np.array(batch_labels), np.array(predictions), encoder, pred_out_dir)
            
            if save_cams:
                utils.cam(model, batch_data, indices, np.array(batch_labels), np.array(predictions), encoder, cam_dir)

            start += len(batch_labels)
            
            if num_batches == total_batches:
                break
        
        h = np.sum(np.array(pred_labels), axis=1)
        idx = np.nonzero(h > 0)[0]
        
        t_labels = encoder.inverse_transform(np.array(test_labels)[idx])
        p_labels = encoder.inverse_transform(np.array(pred_labels)[idx])
        
        print(classification_report(t_labels, p_labels))
        
    def save(self):
        self.pt_model.save(cnt.PT_MODEL_PATH)
        self.color_model.save(cnt.COLOR_MODEL_PATH)
        
    def load(self):
        self.pt_model.load_weights(cnt.PT_MODEL_PATH)
        self.color_model.load_weights(cnt.COLOR_MODEL_PATH)