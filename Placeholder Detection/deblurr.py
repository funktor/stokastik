import data_generator as dg
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, Input
from keras.layers import Dropout, Flatten, Dense, Activation, concatenate, Lambda, Reshape, LeakyReLU, ZeroPadding2D, UpSampling2D, Conv2DTranspose, InputSpec
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
import matplotlib, random
matplotlib.use('agg')
from matplotlib import pyplot as plt
from PIL import Image
from keras.layers.merge import Add
from scipy.ndimage import gaussian_filter
from keras.engine.topology import Layer
import utilities as utils
import tensorflow as tf
from keras.utils import conv_utils

image_shape = (256, 256, 3)
num_random_imgs_display = 10
min_blur_range, max_blur_range = 3, 5

def save_imgs(generator, epoch, full_image_data):
    rand_indices = np.random.randint(0, full_image_data.shape[0], num_random_imgs_display)
    
    real_images = full_image_data[rand_indices]
    blur_images = gaussian_filter(real_images, sigma=(0, random.uniform(min_blur_range, max_blur_range), random.uniform(min_blur_range, max_blur_range), 0))
            
    gen_imgs = generator.predict((blur_images-127.5)/127.5)
    gen_imgs = (gen_imgs * 127.5 + 127.5)/255.0
    
    real_images = real_images/255.0
    blur_images = blur_images/255.0

    fig, axs = plt.subplots(3, num_random_imgs_display, figsize=(256,256))
    
    for j in range(num_random_imgs_display):
        axs[0,j].imshow(real_images[j])
        axs[0,j].axis('off')
        
        axs[1,j].imshow(blur_images[j])
        axs[1,j].axis('off')
        
        axs[2,j].imshow(gen_imgs[j])
        axs[2,j].axis('off')
        
    fig.savefig("images_256/gan_%d.png" % epoch)
    plt.close()
    

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")


class ReflectionPadding2D(Layer):
    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = K.common.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                             padding=self.padding,
                                             data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def res_block(input, filters):
    x = ReflectionPadding2D((1,1))(input)
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(0.5)(x)

    x = ReflectionPadding2D((1,1))(x)
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1))(x)
    x = BatchNormalization()(x)

    merged = Add()([input, x])
    return merged
    

def build_generator():
    inputs = Input(shape=image_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=64, kernel_size=(7,7), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=64*mult*2, kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    mult = 2**n_downsampling
    for i in range(9):
        x = res_block(x, 64*mult)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = Conv2DTranspose(filters=int(64 * mult / 2), kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = ReflectionPadding2D((3,3))(x)
    x = Conv2D(filters=3, kernel_size=(7,7), padding='valid')(x)
    x = Activation('tanh')(x)

    outputs = Add()([x, inputs])
    outputs = Lambda(lambda z: z/2)(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model


def build_discriminator():
    n_layers, use_sigmoid = 3, False
    inputs = Input(shape=image_shape)

    x = Conv2D(filters=64, kernel_size=(4,4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=64*nf_mult, kernel_size=(4,4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=64*nf_mult, kernel_size=(4,4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4,4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    model.compile(loss=wasserstein_loss, optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    
    return model


def build_stacked_model(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_images = generator(inputs)
    discriminator.trainable = False
    outputs = discriminator(generated_images)
    model = Model(inputs=inputs, outputs=[generated_images, outputs])
    model.compile(loss=[perceptual_loss, wasserstein_loss], loss_weights=[100, 1], optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    
    return model

    
def train_gan(num_epochs=500, batch_size=16):
    full_image_data = utils.load_data_npy('all_pt_images_256.npy')
    full_image_data = full_image_data*255.0

    generator = build_generator()
    discriminator = build_discriminator()
    stacked_model = build_stacked_model(generator, discriminator)
    
    n = full_image_data.shape[0]
    
    num_batches, half_batch = int(math.ceil(float(n)/batch_size)), int(batch_size/2)
    
    for epoch in range(num_epochs + 1):
        for batch in range(num_batches):
            discriminator.trainable = True
            
            rand_indices = np.random.randint(0, n, half_batch)
            
            real_images = full_image_data[rand_indices]
            real_labels = np.ones((half_batch, 1))
            
            blur_image_data = gaussian_filter(real_images, sigma=(0, random.uniform(min_blur_range, max_blur_range), random.uniform(min_blur_range, max_blur_range), 0))
            
            fake_images = generator.predict(blur_image_data)
            fake_labels = np.zeros((half_batch, 1))
            
            real_images = (real_images-127.5)/127.5
            fake_images = (fake_images-127.5)/127.5

            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            stacked_model.layers[2] = discriminator
            stacked_model.layers[2].trainable = False
            
            rand_indices = np.random.randint(0, n, batch_size)
            
            real_images = full_image_data[rand_indices]
            blur_image_data = gaussian_filter(real_images, sigma=(0, random.uniform(min_blur_range, max_blur_range), random.uniform(min_blur_range, max_blur_range), 0))
            
            fake_images = generator.predict(blur_image_data)
            
            real_images = (real_images-127.5)/127.5
            fake_images = (fake_images-127.5)/127.5
            
            g_loss = stacked_model.train_on_batch(fake_images, [real_images, np.ones((batch_size, 1))])
            
            generator = stacked_model.layers[1]
            print(epoch, batch, d_loss, g_loss)
            
        save_imgs(generator, epoch, full_image_data)
        
        discriminator.save('discriminator2.h5')
        generator.save('generator2.h5')
        stacked_model.save('stacked_model2.h5')
    
train_gan()
