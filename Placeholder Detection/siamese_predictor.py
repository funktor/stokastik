import data_generator as dg
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, Input
from keras.layers import Dropout, Flatten, Dense, Activation, concatenate, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras.backend as K
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np, glob, math, random
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.utils import shuffle
import collections, keras_metrics as km
from keras.applications.vgg16 import VGG16
import constants as cnt


IMAGE_HEIGHT, IMAGE_WIDTH = cnt.IMAGE_SIZE, cnt.IMAGE_SIZE

def multi_input_generator_streaming(data_generator, batch_size, mode='train', image_data_generator=None):
    if mode == 'train':
        for batch_data, batch_labels in data_generator:
            if image_data_generator is not None:
                image_generator_1 = image_data_generator.flow(batch_data[0], batch_labels, batch_size=batch_size, seed=1)
                image_generator_2 = image_data_generator.flow(batch_data[1], batch_labels, batch_size=batch_size, seed=1)

                gen_1 = image_generator_1.next()
                gen_2 = image_generator_2.next()

                yield [gen_1[0], gen_2[0]], gen_1[1]
            else:
                yield batch_data, batch_labels
            
    
    else:
        for batch_data, batch_labels in data_generator:
            if image_data_generator is not None:
                input_generator_1 = image_data_generator.flow(batch_data[0], batch_size=batch_size, shuffle=False)
                input_generator_2 = image_data_generator.flow(batch_data[1], batch_size=batch_size, shuffle=False)

                gen_1 = input_generator_1.next()
                gen_2 = input_generator_2.next()

                yield [gen_1, gen_2]
            else:
                yield batch_data
                
                
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def custom_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
                

def get_shared_model(image_shape):
    input = Input(shape=image_shape)
    n_layer = input
    
    n_layer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    
    n_layer = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    
    n_layer = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    
    n_layer = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    
    model = Model(inputs=[input], outputs=[n_layer])
    return model
            
            
def get_shared_model_vgg(image_shape):
    input = Input(shape=image_shape)
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input)

    for layer in base_model.layers:
        layer.trainable = False

    n_layer = base_model.output

    n_layer = Flatten()(n_layer)
    
    n_layer = Dense(cnt.EMBEDDING_SIZE)(n_layer)
    n_layer = BatchNormalization()(n_layer)
    
    model = Model(inputs=[input], outputs=[n_layer])
    return model

def my_init(shape, dtype=None):
    return K.constant(1.0/shape[0], shape=(shape[0], 1), name='init_constant')
        
        
class WeightedL2Layer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedL2Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[0][1], 1), initializer=my_init, trainable=True)
        super(WeightedL2Layer, self).build(input_shape)

    def call(self, x):
        return K.sqrt(K.maximum(K.dot(K.square(x[0]-x[1]), K.square(self.kernel)), K.epsilon()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)
    
    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
        

class SiameseModel(object):
    def __init__(self):
        self.classifier = None
        self.model = None
        self.base_model = None
        self.datagen = None
        
        
    def init_classifier(self):
        image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        
        self.datagen = ImageDataGenerator(rotation_range=20,
                                          width_shift_range=0.15, 
                                          height_shift_range=0.15, 
                                          zoom_range=0.1, 
                                          horizontal_flip = True,
                                          vertical_flip = True,
                                          fill_mode='nearest')
        
        input = Input(shape=image_shape)
        
        if cnt.USE_VGG:
            self.base_model = get_shared_model_vgg(image_shape)
        else:
            self.base_model = get_shared_model(image_shape)
            
        n_layer = self.base_model(input)
        
        n_layer = Flatten()(n_layer)
        n_layer = Dense(cnt.EMBEDDING_SIZE, activation='relu')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        
        out = Dense(cnt.NUM_PTS, activation="softmax")(n_layer)
        
        self.classifier = Model(inputs=[input], outputs=[out])

        adam = optimizers.Adam(lr=0.001)
        self.classifier.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
        
        
    def init_model(self):
        image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        
        self.datagen = ImageDataGenerator(rotation_range=20,
                                          width_shift_range=0.15, 
                                          height_shift_range=0.15, 
                                          zoom_range=0.1, 
                                          horizontal_flip = True,
                                          vertical_flip = True,
                                          fill_mode='nearest')
        
        input_a, input_b = Input(shape=image_shape), Input(shape=image_shape)
        
        if cnt.USE_VGG:
            shared_model = get_shared_model_vgg(image_shape)
        else:
            shared_model = get_shared_model(image_shape)
            
        if cnt.PRE_TRAIN_CLASSIFIER:
            self.init_classifier()
            self.classifier.load_weights(cnt.CLASSIFIER_MODEL_PATH)
            
            shared_model.set_weights(self.classifier.layers[1].get_weights())
            
            for layer in shared_model.layers:
                layer.trainable = False
            
        nlayer1 = shared_model(input_a)
        nlayer2 = shared_model(input_b)
        
        n_layer = Flatten()
        nlayer1 = n_layer(nlayer1)
        nlayer2 = n_layer(nlayer2)
        
        n_layer = Dense(cnt.EMBEDDING_SIZE, activation='relu')
        nlayer1 = n_layer(nlayer1)
        nlayer2 = n_layer(nlayer2)
        
        n_layer = BatchNormalization()
        nlayer1 = n_layer(nlayer1)
        nlayer2 = n_layer(nlayer2)
        
        n_layer = Lambda(lambda x: K.l2_normalize(x, axis=1))
        nlayer1 = n_layer(nlayer1)
        nlayer2 = n_layer(nlayer2)
        
        n_layer = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0]-x[1]), axis=1, keepdims=True)))([nlayer1, nlayer2])
        out = Dense(1, activation="sigmoid")(n_layer)

        self.model = Model(inputs=[input_a, input_b], outputs=[out])

        adam = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss="mean_squared_error", metrics=['accuracy', km.precision(label=0), km.recall(label=0)])
        
        
    
    def fit(self):
        if cnt.PRE_TRAIN_CLASSIFIER:
            self.init_classifier()

            callbacks = [
                ModelCheckpoint(filepath=cnt.CLASSIFIER_MODEL_PATH)
            ]

            train_num_batches = int(math.ceil(float(cnt.TRAINING_SAMPLES_PER_EPOCH)/cnt.CLASSIFIER_BATCH_SIZE))

            self.classifier.fit_generator(multi_input_generator_streaming(dg.get_image_data_classifier(cnt.TRAINING_SAMPLES_PER_EPOCH), 
                                                                     cnt.CLASSIFIER_BATCH_SIZE, 
                                                                     mode='train'),
                                     callbacks=callbacks, 
                                     steps_per_epoch=train_num_batches,
                                     epochs=cnt.CLASSIFIER_NUM_EPOCHS, verbose=1)
        
        self.init_model()
        
        callbacks = [
            ModelCheckpoint(filepath=cnt.SIAMESE_BEST_MODEL_PATH, monitor='val_loss', save_best_only=True),
            ModelCheckpoint(filepath=cnt.SIAMESE_MODEL_PATH)
        ]
        
        train_num_batches = int(math.ceil(float(cnt.TRAINING_SAMPLES_PER_EPOCH)/cnt.SIAMESE_BATCH_SIZE))
        valid_num_batches = int(math.ceil(float(cnt.VALIDATION_SAMPLES_PER_EPOCH)/cnt.SIAMESE_BATCH_SIZE))
        
        self.model.fit_generator(multi_input_generator_streaming(dg.get_image_data_siamese(cnt.TRAINING_SAMPLES_PER_EPOCH, 'train'), 
                                                                 cnt.SIAMESE_BATCH_SIZE, 
                                                                 mode='train'),
                                 callbacks=callbacks, 
                                 steps_per_epoch=train_num_batches, 
                                 validation_data=multi_input_generator_streaming(dg.get_image_data_siamese(cnt.VALIDATION_SAMPLES_PER_EPOCH, 'valid'), 
                                                                                 cnt.SIAMESE_BATCH_SIZE, 
                                                                                 mode='train'),
                                 validation_steps=valid_num_batches, 
                                 epochs=cnt.SIAMESE_NUM_EPOCHS, verbose=1)
        
        
    def predict_proba(self, image_data):
        return self.model.predict(image_data)
        
    def predict(self, image_data):
        preds = self.predict_proba(image_data)
        preds = [x[0] for x in preds]
        return np.rint(preds).astype(int)
    
    def score(self):
        data_generator, test_labels, pred_labels = dg.get_image_data_siamese(cnt.VALIDATION_SAMPLES_PER_EPOCH, 'valid'), [], []
        total_batches = int(math.ceil(float(cnt.VALIDATION_SAMPLES_PER_EPOCH)/cnt.SIAMESE_BATCH_SIZE))
        
        num_batches = 0
        for batch_data, batch_labels in data_generator:
            test_labels += batch_labels.tolist()
            pred_labels += self.predict(batch_data).tolist()
            num_batches += 1
            if num_batches == total_batches:
                break
        
        print(classification_report(test_labels, pred_labels))
        
        
    def score_ensemble(self, test_image_arr, voting_images_arr, frac=0.5, probability_threshold=0.5):
        image_data_0, image_data_1 = [], []
        
        for v_img_arr in voting_images_arr:
            image_data_0.append(v_img_arr)
            image_data_1.append(test_image_arr)
            
        image_data_0, image_data_1 = np.array(image_data_0), np.array(image_data_1)
        
        proba = self.predict_proba([image_data_0, image_data_1])
        proba = np.array([x[0] for x in proba])
        
        pred = proba-probability_threshold
        
        pred[pred <= 0] = 0
        pred[pred > 0] = 1
        
        return pred
        
    def save(self):
        self.model.save(cnt.SIAMESE_MODEL_PATH)
        
    def load(self):
        if cnt.USE_BEST_VAL_LOSS_MODEL:
            self.model.load_weights(cnt.SIAMESE_BEST_MODEL_PATH)
        else:
            self.model.load_weights(cnt.SIAMESE_MODEL_PATH)
            
    def get_embeddings(self, X):
        embeddings = K.function([self.model.layers[0].input, self.model.layers[1].input], [self.model.layers[6].get_output_at(0)])
        return embeddings([X, X])[0]
    
    def get_distance_threshold(self):
        weight, bias = self.model.layers[8].get_weights()
        return -float(bias[0]+math.log((1.0/cnt.PLACEHOLDER_THRESHOLD)-1.0))/weight[0][0], weight[0][0]