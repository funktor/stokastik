import keras, os
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, Bidirectional, InputSpec, Lambda, Average, CuDNNLSTM, Flatten, TimeDistributed, Dropout
from keras.models import load_model
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from sklearn.metrics import classification_report
import random, math
from sklearn.model_selection import train_test_split
import data_generator as dg
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import constants as cnt

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
        weights = K.softmax(scores, axis=1)
        weighted_avg = keras.layers.dot([x, weights], axes=1)
        
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
    def __init__(self, vocab_size, training_samples=5000, validation_samples=5000, testing_samples=5000, use_generator=True):
        self.model = None
        self.vocab_size = vocab_size
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.testing_samples = testing_samples
        self.use_generator = use_generator
        
    def init_model(self):
        input1 = Input(shape=(cnt.MAX_WORDS, cnt.WV_EMBEDDING_SIZE))
        input2 = Input(shape=(cnt.MAX_WORDS, cnt.WV_EMBEDDING_SIZE))
        
        nlayer = TimeDistributed(Dense(cnt.SIAMESE_EMBEDDING_SIZE, activation="relu"))
        
        nlayer1 = nlayer(input1)
        nlayer2 = nlayer(input2)
        
        nlayer = Flatten()
        
        nlayer1 = nlayer(nlayer1)
        nlayer2 = nlayer(nlayer2)
        
        nlayer = Dense(cnt.SIAMESE_EMBEDDING_SIZE, activation="linear")
        
        nlayer1 = nlayer(nlayer1)
        nlayer2 = nlayer(nlayer2)
        
        nlayer = Lambda(lambda x: K.l2_normalize(x, axis=1))
        
        nlayer1 = nlayer(nlayer1)
        nlayer2 = nlayer(nlayer2)
        
        merge = Lambda(lambda x: K.sqrt(K.maximum(K.sum(K.square(x[0]-x[1]), axis=1, keepdims=True), K.epsilon())))([nlayer1, nlayer2])
        out = Dense(1, activation="sigmoid")(merge)

        self.model = Model([input1, input2], out)
        adam = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        
    def fit(self):
        self.init_model()
        
        callbacks = [
            ModelCheckpoint(filepath=os.path.join(cnt.DATA_FOLDER, cnt.SIAMESE_MODEL_FILE), monitor='val_loss', save_best_only=True),
        ]
        
        if self.use_generator:
            train_num_batches = int(math.ceil(float(self.training_samples)/cnt.SIAMESE_BATCH_SIZE))
            valid_num_batches = int(math.ceil(float(self.validation_samples)/cnt.SIAMESE_BATCH_SIZE))


            self.model.fit_generator(dg.get_data_as_generator(self.training_samples, 'train'),
                                     callbacks=callbacks, 
                                     steps_per_epoch=train_num_batches, 
                                     validation_data=dg.get_data_as_generator(self.validation_samples, 'validation'),
                                     validation_steps=valid_num_batches, 
                                     epochs=cnt.SIAMESE_NUM_EPOCHS, verbose=1, use_multiprocessing=True)
        else:
            X_train, y_train = dg.get_data_as_vanilla(self.training_samples, 'train')
            X_valid, y_valid = dg.get_data_as_vanilla(self.validation_samples, 'validation')
            
            self.model.fit(X_train, y_train, 
                           batch_size=cnt.SIAMESE_BATCH_SIZE, 
                           validation_data=(X_valid, y_valid), 
                           callbacks=callbacks, 
                           epochs=cnt.SIAMESE_NUM_EPOCHS, verbose=1, shuffle=True)
        
    def save(self):
        self.model.save(os.path.join(cnt.DATA_FOLDER, cnt.SIAMESE_MODEL_FILE))
        
    def load(self):
        self.model.load_weights(os.path.join(cnt.DATA_FOLDER, cnt.SIAMESE_MODEL_FILE))
    
    def predict_proba(self, test_data):
        return self.model.predict(test_data)
        
    def predict(self, test_data):
        return np.rint(self.predict_proba(test_data)).astype(int)
    
    def score(self):
        if self.use_generator:
            data_generator, test_labels, pred_labels = dg.get_data_as_generator(self.testing_samples, 'test'), [], []
            total_batches = int(math.ceil(float(self.testing_samples)/cnt.SIAMESE_BATCH_SIZE))

            num_batches = 0
            for batch_data, batch_labels in data_generator:
                test_labels += batch_labels.tolist()
                pred_labels += self.predict(batch_data).tolist()
                num_batches += 1
                if num_batches == total_batches:
                    break
        else:
            X_test, test_labels = dg.get_data_as_vanilla(self.testing_samples, 'test')
            pred_labels = self.predict(X_test)
        
        print(classification_report(test_labels, pred_labels))
    
    def get_embeddings(self, X):
        embeddings = K.function([self.model.layers[0].input, self.model.layers[1].input], [self.model.layers[5].get_output_at(0)])
        return embeddings([X, X])[0]