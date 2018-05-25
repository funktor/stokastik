% matplotlib inline

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Flatten, Reshape, LSTM, RepeatVector, \
    Dropout
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam
import cv2, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical


def get_images_from_folder(folder):
    images, labels = [], []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label = filename.split('.')[0]

        if img is not None:
            images.append(img)

            if label == 'cat':
                labels.append(0)
            else:
                labels.append(1)

    labels = np.array(labels)
    labels = labels.reshape((len(labels), 1))

    return np.array(images), labels


def get_input_data(folder):
    mydata, labels = get_images_from_folder(folder)
    mydata = mydata.astype('float32') / 255
    mydata = mydata.reshape((len(mydata), mydata.shape[1], mydata.shape[2], 1))

    return mydata, labels


def load_mnist_data():
    (X_train, train_labels), (X_test, test_labels) = mnist.load_data()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train = X_train.reshape((len(X_train), X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((len(X_test), X_test.shape[1], X_test.shape[2], 1))

    return X_train, X_test, train_labels, test_labels


def standard_autoencoder(X_train):
    m = np.prod(X_train.shape[1:])
    input_img = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    # Image needs to be flattened in order to train with Dense layers
    x = Flatten()(input_img)

    # Encoder
    x = Dense(512, activation='relu')(x)
    encoded = Dense(256, activation='relu')(x)
    encoder = Model(input_img, encoded)

    # Decoder
    decoder_input = Input(shape=(256,))
    x = Dense(512, activation='relu')(decoder_input)
    x = Dense(m, activation='sigmoid')(x)
    decoded = Reshape((X_train.shape[1], X_train.shape[2], X_train.shape[3]))(x)
    decoder = Model(decoder_input, decoded)

    # Full Autoencoder
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    adam = Adam(lr=0.0005)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    autoencoder.fit(X_train, X_train, epochs=10, shuffle=True, batch_size=32)

    return encoder, decoder, autoencoder


def variational_autoencoder(X_train):
    m = np.prod(X_train.shape[1:])
    input_img = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    # Image needs to be flattened in order to train with Dense layers
    x = Flatten()(input_img)

    # Encoder
    x = Dense(512, activation='relu')(x)
    z_mean, z_sigma = Dense(256)(x), Dense(256)(x)
    z = Lambda(sampling, output_shape=(256,))([z_mean, z_sigma])
    encoder = Model(input_img, [z_mean, z_sigma, z])

    # Decoder
    decoder_input = Input(shape=(256,))
    x = Dense(512, activation='relu')(decoder_input)
    x = Dense(m, activation='sigmoid')(x)
    decoded = Reshape((X_train.shape[1], X_train.shape[2], X_train.shape[3]))(x)
    decoder = Model(decoder_input, decoded)

    # Full Autoencoder
    autoencoder_out = decoder(encoder(input_img)[2])
    autoencoder = Model(input_img, autoencoder_out)

    # Loss is summation of reconstruction loss and KL Loss
    reconstruction_loss = K.sum(K.binary_crossentropy(Flatten()(input_img), Flatten()(autoencoder_out)), axis=-1)
    kl_loss = - 0.5 * K.sum(1.0 + z_sigma - K.square(z_mean) - K.exp(z_sigma), axis=-1)

    autoencoder.add_loss(K.mean(reconstruction_loss + kl_loss))
    adam = Adam(lr=0.0005)
    autoencoder.compile(optimizer=adam, loss=None)
    autoencoder.fit(X_train, shuffle=True, epochs=100, batch_size=32)

    return encoder, decoder, autoencoder


def convolution_autoencoder(X_train):
    input_img = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    enc_out = Flatten()(x)
    encoder = Model(input_img, enc_out)

    # Decoder
    decoder_input = Input(shape=(196,))
    p = Reshape((14, 14, 1))(decoder_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(p)
    x = UpSampling2D((2, 2))(x)
    dec_out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_input, dec_out)

    # Full Autoencoder
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    adam = Adam(lr=0.0005)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, shuffle=True)

    return encoder, decoder, autoencoder


def sampling(args):
    z_mean, z_sigma = args
    batch, dim = K.shape(z_mean)[0], K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=0.1)

    return z_mean + K.exp(0.5 * z_sigma) * epsilon


def conv_variational_autoencoder(X_train):
    m = np.prod(X_train.shape[1:])
    input_img = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    q = Flatten()(x)
    z_mean, z_sigma = Dense(196)(q), Dense(196)(q)
    z = Lambda(sampling, output_shape=(196,))([z_mean, z_sigma])
    encoder = Model(input_img, [z_mean, z_sigma, z])

    # Decoder
    decoder_input = Input(shape=(196,))
    p = Reshape((14, 14, 1))(decoder_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(p)
    x = UpSampling2D((2, 2))(x)
    dec_out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_input, dec_out)

    # Full Autoencoder
    autoencoder_out = decoder(encoder(input_img)[2])
    autoencoder = Model(input_img, autoencoder_out)

    # Loss is summation of reconstruction loss and KL Loss
    reconstruction_loss = K.sum(K.binary_crossentropy(Flatten()(input_img), Flatten()(autoencoder_out)), axis=-1)
    kl_loss = - 0.5 * K.sum(1.0 + z_sigma - K.square(z_mean) - K.exp(z_sigma), axis=-1)

    autoencoder.add_loss(K.mean(reconstruction_loss + kl_loss))
    adam = Adam(lr=0.0005)
    autoencoder.compile(optimizer=adam, loss=None)
    autoencoder.fit(X_train, shuffle=True, epochs=10, batch_size=32)

    return encoder, decoder, autoencoder


def train_classifier(encoded_imgs, labels):
    inputs = Input(shape=(encoded_imgs.shape[1],))

    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(encoded_imgs, labels, epochs=50, batch_size=32, shuffle=True)

    return model


def save_model(model, model_json_path, model_wts_path):
    model_json = model.to_json()

    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_wts_path)


def load_model(model_json_path, model_wts_path):
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(model_wts_path)

    return loaded_model


def get_representation(encoder, X_test):
    encoded = encoder.predict(X_test)
    encoded = encoded.reshape((len(encoded), np.prod(encoded.shape[1:])))

    return encoded


# X_train, X_test, train_labels, test_labels = load_mnist_data()
# (X_train, _), (X_test, _) = mnist.load_data()
# print X_train.shape

# X_train, train_labels = get_input_data("/Users/funktor/Downloads/cats")
# print X_train.shape

# X_test, test_labels = get_input_data("/Users/funktor/Downloads/cats_test")
# print X_test.shape

# with open("/Users/funktor/train_data.pkl", "wb") as train_f:
#     pickle.dump(X_train, train_f)

# with open("/Users/funktor/test_data.pkl", "wb") as test_f:
#     pickle.dump(X_test, test_f)

# encoder, decoder, autoencoder = conv_variational_autoencoder(X_train[:5000])

# encoder = load_model("encoder.json", "encoder.h5")
# decoder = load_model("decoder.json", "decoder.h5")
# autoencoder = load_model("autoencoder.json", "autoencoder.h5")

# encoded_img = encoder.predict(X_train[5:6])
# encoded_img = encoded_img.reshape((len(encoded_img), 32, 32, 1))

# decoded_img = decoder.predict(encoded_img)

# save_model(encoder, "encoder.json", "encoder.h5")
# save_model(autoencoder, "autoencoder.json", "autoencoder.h5")

# encoded_imgs = encoder.predict(X_train[:20000])
# encoded_imgs = encoded_imgs + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=encoded_imgs.shape)
# encoded_imgs = encoded_imgs[2]
# w = 0.5 * (encoded_imgs[0:1] + encoded_imgs[1:2])
# decoded_imgs = decoder.predict(encoded_imgs)

# plt.imshow(X_train[11].reshape(28, 28))
# plt.show()

# for i in range(10):
#     encoded_img = encoder.predict(X_train[11:12])
#     encoded_img = encoded_img[2]
#     decoded_img = decoder.predict(encoded_img)
#     plt.imshow(decoded_img[0].reshape(28, 28))
#     plt.show()

# model = train_classifier(encoded_imgs, to_categorical(train_labels)[:20000])
# enc_imgs = encoded_imgs[2]
# enc_imgs = enc_imgs.reshape((len(enc_imgs), np.prod(enc_imgs.shape[1:])))

# encoded_imgs = encoder.predict(X_test[:20000])
# encoded_imgs = encoded_imgs[2]
# model.evaluate(encoded_imgs, to_categorical(test_labels)[:20000])

# print len(encoded_imgs)

# w = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
# kmeans = KMeans(n_clusters=2).fit(enc_imgs)

# print kmeans.labels_[:200]

# decoded_imgs = autoencoder.predict(X_test[:10])

# for i in range(10):
#     plt.imshow(X_test[i].reshape(28, 28))
#     plt.show()
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.show()