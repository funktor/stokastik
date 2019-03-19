import data_generator as dg
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, Input
from keras.layers import Dropout, Flatten, Dense, Activation, concatenate, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np, glob, math, keras_metrics as km
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

def input_generator_streaming(data_generator, batch_size, mode='train', image_data_generator=None):
    if mode == 'train':
        for batch_data, batch_labels in data_generator:
            if image_data_generator is not None:
                image_generator = image_data_generator.flow(batch_data, batch_labels, batch_size=batch_size)
                gen = image_generator.next()
                yield gen[0], gen[1]
            else:
                yield batch_data, batch_labels
            
    
    else:
        for batch_data, batch_labels in data_generator:
            if image_data_generator is not None:
                input_generator = image_data_generator.flow(batch_data, batch_size=batch_size, shuffle=False)
                gen = image_generator.next()
                yield gen
            else:
                yield batch_data
                
                
def get_model(image_shape):
    input = Input(shape=image_shape)
    n_layer = input
    
    for i in range(3, 8):
        n_layer = Conv2D(filters=2**i, kernel_size=(3, 3), padding='same', activation='relu')(n_layer)
        n_layer = BatchNormalization()(n_layer)

        n_layer = Conv2D(filters=2**i, kernel_size=(3, 3), padding='same', activation='relu')(n_layer)
        n_layer = BatchNormalization()(n_layer)

        n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)

    n_layer = Flatten()(n_layer)

    n_layer = Dense(4096, activation='relu')(n_layer)
    n_layer = Dropout(0.5)(n_layer)
    n_layer = BatchNormalization()(n_layer)
    
    model = Model(inputs=[input], outputs=[n_layer])
    return model
            
            
def get_model_vgg(image_shape):
    input = Input(shape=image_shape)
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input)

    for layer in base_model.layers:
        layer.trainable = False

    n_layer = base_model.output

    n_layer = Flatten()(n_layer)
    
<<<<<<< HEAD
    n_layer = Dense(2048, activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    
    n_layer = Dense(1024, activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)

    n_layer = Dense(512, activation='relu')(n_layer)
=======
    n_layer = Dense(4096, activation='relu')(n_layer)
    n_layer = Dropout(0.5)(n_layer)
    n_layer = BatchNormalization()(n_layer)
    
    n_layer = Dense(32, activation='relu')(n_layer)
>>>>>>> 482d78c93b9f96d6b155fc2e6de49e3d5354da1c
    n_layer = BatchNormalization()(n_layer)
    
    model = Model(inputs=[input], outputs=[n_layer])
    return model


class UniModel(object):
    def __init__(self, model_file_path, best_model_file_path, batch_size=256, training_samples=5000, validation_samples=5000, testing_samples=5000, use_vgg=True):
        self.model = None
        self.datagen = None
        self.model_file_path = model_file_path
        self.best_model_file_path = best_model_file_path
        self.batch_size = batch_size
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.use_vgg = use_vgg
        self.testing_samples = testing_samples
        
    def init_model(self):
        image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        
        self.datagen = ImageDataGenerator(rotation_range=20,
                                          width_shift_range=0.15, 
                                          height_shift_range=0.15, 
                                          zoom_range=0.1, 
                                          horizontal_flip = True,
                                          vertical_flip = True,
                                          fill_mode='nearest')
        
        if self.use_vgg:
            model = get_model_vgg(image_shape)
        else:
            model = get_model(image_shape)
        
        input = Input(shape=image_shape)
        n_layer = model(input)
        out = Dense(1, activation="sigmoid")(n_layer)
        
        self.model = Model(inputs=[input], outputs=[out])

        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy', km.precision(), km.recall()])
    
    def fit(self):
        self.init_model()
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=3),
            ModelCheckpoint(filepath=self.best_model_file_path, monitor='val_loss', save_best_only=True),
            ModelCheckpoint(filepath=self.model_file_path)
        ]
        
        train_num_batches = int(math.ceil(float(self.training_samples)/self.batch_size))
<<<<<<< HEAD
        valid_num_batches = int(math.ceil(float(self.testing_samples)/self.batch_size))
=======
        valid_num_batches = int(math.ceil(float(self.validation_samples)/self.batch_size))
>>>>>>> 482d78c93b9f96d6b155fc2e6de49e3d5354da1c
        
        self.model.fit_generator(input_generator_streaming(dg.get_image_data_unimodel(self.training_samples, 
                                                                                      'train', 
                                                                                      batch_size=self.batch_size), 
                                                           self.batch_size, 
                                                           mode='train',
                                                           image_data_generator=self.datagen),
                                 callbacks=callbacks, 
                                 steps_per_epoch=train_num_batches, 
<<<<<<< HEAD
                                 validation_data=input_generator_streaming(dg.get_image_data_unimodel(self.testing_samples, 
                                                                                                      'test', 
=======
                                 validation_data=input_generator_streaming(dg.get_image_data_unimodel(self.validation_samples, 
                                                                                                      'train', 
>>>>>>> 482d78c93b9f96d6b155fc2e6de49e3d5354da1c
                                                                                                      batch_size=self.batch_size), 
                                                                           self.batch_size, 
                                                                           mode='train'),
                                 validation_steps=valid_num_batches, 
                                 epochs=10, verbose=1)
        
    def predict_proba(self, image_data):
        return self.model.predict(image_data)
        
    def predict(self, image_data):
        return np.rint(self.predict_proba(image_data)).astype(int)
    
    def score(self):
        data_generator, test_labels, pred_labels = dg.get_image_data_unimodel(self.testing_samples, 'test', batch_size=self.batch_size), [], []
        total_batches = int(math.ceil(float(self.testing_samples)/self.batch_size))
        
        num_batches = 0
        for batch_data, batch_labels in data_generator:
            test_labels += batch_labels.tolist()
            pred_labels += self.predict(batch_data).tolist()
            num_batches += 1
            if num_batches == total_batches:
                break
        
        print(classification_report(test_labels, pred_labels))
        
    def save(self):
        self.model.save(self.model_file_path)
        
    def load(self, best_model=False):
        if best_model:
            self.model.load_weights(self.best_model_file_path)
        else:
            self.model.load_weights(self.model_file_path)