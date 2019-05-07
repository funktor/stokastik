import data_generator as dg
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, Input
from keras.layers import Dropout, Flatten, Dense, Activation, concatenate, Lambda, Reshape, LeakyReLU, ZeroPadding2D, UpSampling2D
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
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from PIL import Image

def save_imgs(generator, epoch, noise_shape):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, noise_shape))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/gan_%d.png" % epoch)
    plt.close()

def build_generator(noise_shape=(128,)):
    input = Input(noise_shape)
    x = Dense(128 * 16 * 16, activation="relu")(input)
    x = Reshape((16, 16, 128))(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(3, kernel_size=3, padding="same")(x)
    out = Activation("tanh")(x)
    
    model = Model(input, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    return model

def build_discriminator(img_shape=(64,64,3)):
    input = Input(img_shape)
    
    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    x = (LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(input, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    
    return model

def build_stacked_model(generator, discriminator, noise_shape):
    noise_input = Input(shape=noise_shape)
    generated_img = generator(noise_input)
    discriminator.trainable = False
    discriminator_out = discriminator(generated_img)
    stacked_model = Model(noise_input, discriminator_out)
    
    stacked_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    return stacked_model
    
def train_gan(num_epochs=10000, batch_size=32):
    image_data = dg.load_data_npy('train_siamese_image_data.npy')
    img_shape = (image_data.shape[1], image_data.shape[2], image_data.shape[3])
    print("Shape=", img_shape)
    noise_shape=(100,)
    
    discriminator = build_discriminator(img_shape)
    generator = build_generator(noise_shape)
    stacked_model = build_stacked_model(generator, discriminator, noise_shape)
    
    n = image_data.shape[0]
    
    num_batches, half_batch = int(math.ceil(float(n)/batch_size)), int(batch_size/2)
    
    for epoch in range(num_epochs + 1):
        for batch in range(num_batches):
            discriminator.trainable = True
            
            noise = np.random.normal(0, 1, (half_batch, noise_shape[0]))
            
            fake_images = generator.predict(noise)
            fake_labels = np.zeros((half_batch, 1))
            
            rand_indices = np.random.randint(0, n, half_batch)
            real_images = image_data[rand_indices]
            real_labels = np.ones((half_batch, 1))

            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, (batch_size, noise_shape[0]))
            
            stacked_model.layers[2] = discriminator
            stacked_model.layers[2].trainable = False
            
            g_loss = stacked_model.train_on_batch(noise, np.ones((batch_size, 1)))
            
            generator = stacked_model.layers[1]
            print("Epoch %d Batch %d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(epoch, batch, num_batches, d_loss[0], noise_shape[0] * d_loss[1], g_loss))
        save_imgs(generator, epoch, noise_shape[0])
        
        discriminator.save('discriminator.h5')
        generator.save('generator.h5')
        stacked_model.save('stacked_model.h5')
    
train_gan()
