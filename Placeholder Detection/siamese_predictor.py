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
import numpy as np, glob, math, random
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.utils import shuffle
import collections, keras_metrics as km
from keras.applications.vgg16 import VGG16


IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

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
                

def get_shared_model(image_shape):
    input = Input(shape=image_shape)
    n_layer = input
    
    n_layer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    
    n_layer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    
    n_layer = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)

    n_layer = Flatten()(n_layer)
    
    n_layer = Dense(4096, activation='linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    
    model = Model(inputs=[input], outputs=[n_layer])
    return model
            
            
def get_shared_model_vgg(image_shape):
    input = Input(shape=image_shape)
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input)

    for layer in base_model.layers:
        layer.trainable = False

    n_layer = base_model.output

    n_layer = Flatten()(n_layer)
    
    n_layer = Dense(4096, activation='linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    
    model = Model(inputs=[input], outputs=[n_layer])
    return model
        

class SiameseModel(object):
    def __init__(self, model_file_path, best_model_file_path, batch_size=256, training_samples=5000, validation_samples=5000, testing_samples=5000, use_vgg=False):
        self.model = None
        self.datagen = None
        self.model_file_path = model_file_path
        self.best_model_file_path = best_model_file_path
        self.batch_size = batch_size
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.testing_samples = testing_samples
        self.use_vgg = use_vgg
        
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
        
        if self.use_vgg:
            shared_model = get_shared_model_vgg(image_shape)
        else:
            shared_model = get_shared_model(image_shape)
            
        shared_model_a, shared_model_b = shared_model(input_a), shared_model(input_b)
        
        n_layer = Lambda(lambda x: K.sqrt(K.maximum(K.sum(K.square(x[0]-x[1]), axis=1, keepdims=True), K.epsilon())))([shared_model_a, shared_model_b])
        n_layer = BatchNormalization()(n_layer)
        
        out = Dense(1, activation="sigmoid")(n_layer)

        self.model = Model(inputs=[input_a, input_b], outputs=[out])

        adam = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy', km.precision(label=0), km.recall(label=0)])
    
    def fit(self):
        self.init_model()
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=3),
            ModelCheckpoint(filepath=self.best_model_file_path, monitor='val_loss', save_best_only=True),
            ModelCheckpoint(filepath=self.model_file_path)
        ]
        
        train_num_batches = int(math.ceil(float(self.training_samples)/self.batch_size))
        valid_num_batches = int(math.ceil(float(self.validation_samples)/self.batch_size))
        
        self.model.fit_generator(multi_input_generator_streaming(dg.get_image_data_siamese(self.training_samples, 
                                                                                           'train', 
                                                                                           batch_size=self.batch_size), 
                                                                 self.batch_size, 
                                                                 mode='train'),
                                 callbacks=callbacks, 
                                 steps_per_epoch=train_num_batches, 
                                 validation_data=multi_input_generator_streaming(dg.get_image_data_siamese(self.validation_samples,
                                                                                                           'valid', 
                                                                                                           batch_size=self.batch_size), 
                                                                                 self.batch_size, 
                                                                                 mode='train'),
                                 validation_steps=valid_num_batches, 
                                 epochs=5, verbose=1)
        
        
    def predict_proba(self, image_data):
        return self.model.predict(image_data)
        
    def predict(self, image_data):
        return np.rint(self.predict_proba(image_data)).astype(int)
    
    def score(self):
        data_generator, test_labels, pred_labels = dg.get_image_data_siamese(self.testing_samples, 'test', batch_size=self.batch_size), [], []
        total_batches = int(math.ceil(float(self.testing_samples)/self.batch_size))
        
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
        index_map = collections.defaultdict(list)
        
        for v_img_arr in voting_images_arr:
            image_data_0.append(v_img_arr)
            image_data_1.append(test_image_arr)
            
        image_data_0, image_data_1 = np.array(image_data_0), np.array(image_data_1)
        
        proba = self.predict_proba([image_data_0, image_data_1])
        proba = np.array([x[0] for x in proba])
        
        pred = proba-probability_threshold
        
        pred[pred <= 0] = 0
        pred[pred > 0] = 1
        
        return 0 if np.count_nonzero(pred) < frac*voting_images_arr.shape[0] else 1
        
    def save(self):
        self.model.save(self.model_file_path)
        
    def load(self, best_model=False):
        if best_model:
            self.model.load_weights(self.best_model_file_path)
        else:
            self.model.load_weights(self.model_file_path)