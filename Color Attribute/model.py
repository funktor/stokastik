import keras, os, sys
sys.path.append('/home/jupyter/stormbreaker/')

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, Bidirectional, InputSpec, Lambda, Average, CuDNNLSTM, Flatten, TimeDistributed, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import load_model
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from sklearn.metrics import classification_report
import random, math, pickle, tables
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.applications.vgg16 import VGG16
import collections, keras_metrics as km
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
import tables, pandas as pd, os, pickle
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy, json
import placeholder_detection.utilities as utils
from skimage.color import rgb2lab, deltaE_cie76

batch_size = 64
image_size = 128

rgb_values = [(0,0,0,'black'), (255,255,255,'white'), (255,0,0,'red'), (0,0,255,'blue'), (255,255,0,'yellow'), 
              (192,192,192,'silver'), (128,128,128,'grey'), (0,255,0,'green'), (128,0,128,'purple'), (0,0,128,'navy'), 
              (165,42,42,'brown'), (255,165,0,'orange'), (255,215,0,'gold'), (64,224,208,'turquoise'), 
              (255,192,203, 'pink'), (245,245,220,'beige')]


def url_type_fn(row):
    a = json.loads(row['image_urls'])
    b = json.loads(row['image_url_types'])
    if "PRIMARY" in b:
        i = b.index("PRIMARY")
        c = a[i]
    else:
        c = a[1] if len(a) > 1 else a[0]
    row['image_urls'] = c
    return row

def read_input_file(input_file_path='color_ae.tsv'):
    df = pd.read_csv(input_file_path, sep='\\t', engine='python')
    df = df.dropna()
    df = df.groupby('color_category').filter(lambda x : len(x)>=100)
    df.color_category.replace(to_replace = "Multi-color", value ="Multicolor", inplace=True)
    df.drop_duplicates(subset=['item_id'], inplace=True)
    df = df.apply(lambda row: url_type_fn(row), axis=1)
    urls = list(df.image_urls)
    filenames = [os.path.join('color_images', utils.url_to_filename(url)) for url in urls]
    df['image_path'] = filenames
    
    with open('urls_list.txt', 'w') as f:
        for url in urls:
            f.write(url+'\n')
    f.close()
    df.to_csv('color_urls.csv', sep=",", encoding='utf-8')
    

def download_images(urls_file_path='urls_list.txt'):
    utils.fetch_urls_parallel(urls_file_path, 'color_images', 6)
    
def convert_to_rgb(images_dir='color_images', num_threads=5):
    pool = ThreadPool(num_threads)
    http_codes = pool.map(lambda img_file: Image.open(img_file).convert('RGB').save(img_file), os.listdir(images_dir))
    pool.close()
    pool.join()

def save_data_npy(data, path):
    np.save(os.path.join("data", path), data)
    
def load_data_npy(path):
    return np.load(os.path.join("data", path))
    
def save_data_pkl(data, path):
    with open(os.path.join("data", path), 'wb') as f:
        pickle.dump(data, f)
        
def load_data_pkl(path):
    with open(os.path.join("data", path), 'rb') as f:
        return pickle.load(f)
    
    
def create_image_data():
    try:
        img_arr_file = tables.open_file('data/color_attribute_image_array.h5', mode='w')
        atom = tables.Float32Atom()
        img_arr = img_arr_file.create_earray(img_arr_file.root, 'data', atom, (0, image_size, image_size, 3))

        chunk_size, pt_labels, color_labels = 5000, [], []
        for df_chunk in pd.read_csv('color_urls.csv', chunksize=chunk_size):
            df = df_chunk[list(df_chunk['image_path'].apply(lambda x:os.path.exists(x)))]
            print(df.shape)
            pt_labels += list(df['product_type'])
            color_labels += list(df['color_category'])
            file_paths = list(df['image_path'])
            img_arr.append([img_to_array(load_img(image).convert('RGB').resize((image_size, image_size)))/255.0 for image in file_paths])
            
        save_data_pkl(pt_labels, 'product_type_attribute_labels.pkl')
        save_data_pkl(color_labels, 'color_attribute_labels.pkl')
    finally:
        img_arr_file.close()
        
    
def create_data_pairs():
    try:
        img_arr_file = tables.open_file('data/color_attribute_image_array.h5', mode='r')
        img_arr = img_arr_file.root.data
        
        train_indices, test_indices = train_test_split(range(img_arr.shape[0]), test_size=0.15)
        
        color_encoder = LabelBinarizer()
        pt_encoder = LabelBinarizer()
        
        color_labels = load_data_pkl('color_attribute_labels.pkl')
        transfomed_color_labels = color_encoder.fit_transform(color_labels)
        
        pt_labels = load_data_pkl('product_type_attribute_labels.pkl')
        transfomed_pt_labels = pt_encoder.fit_transform(pt_labels)
        
        save_data_pkl(transfomed_color_labels, 'transfomed_color_labels.pkl')
        save_data_pkl(transfomed_pt_labels, 'transfomed_pt_labels.pkl')
        save_data_pkl(pt_encoder, 'pt_encoder.pkl')
        save_data_pkl(color_encoder, 'color_encoder.pkl')
        
        print(len(train_indices), len(test_indices))
        
        save_data_pkl(train_indices, 'train_indices.pkl')
        save_data_pkl(test_indices, 'test_indices.pkl')
            
    finally:
        img_arr_file.close()
        
        
def get_data_as_generator(num_samples, batch_size=64, prefix='train', type='pt'):
    try:
        img_arr_file = tables.open_file('data/color_attribute_image_array.h5', mode='r')
        img_arr = img_arr_file.root.data
        
        if type == 'pt':
            labels = load_data_pkl('transfomed_pt_labels.pkl')
        else:
            labels = load_data_pkl('transfomed_color_labels.pkl')
            
        labels = np.array(labels)
        
        random.seed(42)
        
        indices = load_data_pkl(prefix + '_indices.pkl')
        
        random.shuffle(indices)
        indices = np.array(indices)

        n = min(num_samples, len(indices))
        num_batches = int(math.ceil(float(n)/batch_size))

        batch_num = 0

        while True:
            m = batch_num % num_batches

            start, end = m*batch_size, min((m+1)*batch_size, n)
            batch_indices = indices[start:end]
            
            out_img_arr = np.array([img_arr[x] for x in batch_indices])
            
            batch_num += 1

            yield out_img_arr, labels[batch_indices]
            
    finally:
        img_arr_file.close()
        
        
def cnn_model():
    input = Input(shape=(image_size, image_size, 3))
    n_layer = input
    
    n_layer = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(n_layer)
    #(128, 128, 64)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    #(64, 64, 64)
    
    n_layer = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(n_layer)
    #(64, 64, 128)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    #(32, 32, 128)
    
    n_layer = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(n_layer)
    #(32, 32, 128)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    #(16, 16, 128)
    
    n_layer = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(n_layer)
    #(16, 16, 256)
    n_layer = BatchNormalization()(n_layer)
    n_layer = MaxPooling2D(pool_size=(2, 2))(n_layer)
    #(8, 8, 256)
    
    n_layer = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(n_layer)
    #(8, 8, 256)
    n_layer = BatchNormalization()(n_layer)
    
    model = Model(inputs=input, outputs=n_layer)
    return model

        
def init_pt_model():
    input = Input(shape=(image_size, image_size, 3))
    n_layer = cnn_model()(input)
    
    n_layer = AveragePooling2D(pool_size=(8, 8))(n_layer)
    #(1, 1, 256)

    n_layer = Flatten()(n_layer)
    n_layer = Dropout(0.25)(n_layer)
    n_layer = BatchNormalization()(n_layer)
    
    out = Dense(24, activation="softmax")(n_layer)

    model = Model(inputs=input, outputs=out)

    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
    
    return model


def init_color_model():
    input = Input(shape=(image_size, image_size, 3))
    cnn = cnn_model()
    
    pt_model = init_pt_model()
    pt_model.load_weights('data/pt_model.h5')
    
    cnn.set_weights(pt_model.layers[1].get_weights())

    n_layer = cnn(input)
    
    n_layer = AveragePooling2D(pool_size=(8, 8))(n_layer)
    #(1, 1, 256)
    
    n_layer = Flatten()(n_layer)
    n_layer = Dropout(0.25)(n_layer)
    n_layer = BatchNormalization()(n_layer)
    
    out = Dense(20, activation="softmax")(n_layer)

    model = Model(inputs=input, outputs=out)

    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
    
    return model

    
def train_model():
    train_indices = load_data_pkl('train_indices.pkl')
    test_indices = load_data_pkl('test_indices.pkl')
    
    train_num_batches = int(math.ceil(float(len(train_indices))/batch_size))
    valid_num_batches = int(math.ceil(float(len(test_indices))/batch_size))
    
    pt_model = init_pt_model()
    
    callbacks = [
        ModelCheckpoint(filepath='data/pt_model.h5', monitor='val_loss', save_best_only=True),
    ]

    pt_model.fit_generator(get_data_as_generator(len(train_indices), batch_size, 'train', 'pt'),
                        callbacks=callbacks,
                        steps_per_epoch=train_num_batches, 
                        validation_data=get_data_as_generator(len(test_indices), batch_size, 'test', 'pt'),
                        validation_steps=valid_num_batches, 
                        epochs=20, verbose=2, use_multiprocessing=True)
    
    color_model = init_color_model()
    
    callbacks = [
        ModelCheckpoint(filepath='data/color_model.h5', monitor='val_loss', save_best_only=True),
    ]

    color_model.fit_generator(get_data_as_generator(len(train_indices), batch_size, 'train', 'color'),
                        callbacks=callbacks,
                        steps_per_epoch=train_num_batches, 
                        validation_data=get_data_as_generator(len(test_indices), batch_size, 'test', 'color'),
                        validation_steps=valid_num_batches, 
                        epochs=20, verbose=2, use_multiprocessing=True)
    
    
def color_distance(rgb1, rgb2):
    rm = 0.5*(rgb1[0]+rgb2[0])
    d = sum((2.0+rm/256.0, 4.0, 2.0+(255.0-rm)/256.0)*(rgb1-rgb2)**2)**0.5
    return d

def get_predicted_colors(actual_img, heat_map):
    ind = np.unravel_index(np.argsort(heat_map, axis=None)[::-1], heat_map.shape)
    rgb_preds = zip(actual_img[ind], heat_map[ind])
    color_wts, color_cnts = dict(), dict()

    for z, u in rgb_preds:
        r, g, b = z
        min_dist, best_col = float("Inf"), None
        for r1, g1, b1, col in rgb_values:
            dist = color_distance(np.array([r,g,b]), np.array([r1,g1,b1]))
            if dist < min_dist:
                min_dist = dist
                best_col = col

        if best_col not in color_wts:
            color_wts[best_col] = 0
        color_wts[best_col] += u

        if best_col not in color_cnts:
            color_cnts[best_col] = 0
        color_cnts[best_col] += 1

    color_wts = {k:float(v)/color_cnts[k] for k, v in color_wts.items()}

    colors = [(v, k) for k, v in color_wts.items()]
    colors = sorted(colors, key=lambda k:-k[0])[:5]
    colors = [c for d, c in colors]
    
    return colors

    
def cam(model, image_array, indices, true_labels, pred_labels, encoder, out_dir):
    image_array = np.array(image_array*255, dtype=np.uint8)
    
    class_weights = model.layers[-1].get_weights()[0]
    
    get_last_conv_output = K.function([model.layers[0].input, model.layers[1].get_input_at(0)], [model.layers[1].get_output_at(0)])
    
    conv_outputs = get_last_conv_output([image_array, image_array])[0]
    conv_outputs = scipy.ndimage.zoom(conv_outputs, (1, 16, 16, 1), order=1)
    
    for idx in range(len(indices)):
        index = indices[idx]
        
        t_label = encoder.inverse_transform(true_labels[idx:idx+1])[0]
        p_label = encoder.inverse_transform(pred_labels[idx:idx+1])[0]
        
        a = np.nonzero(true_labels[idx])
        
        if len(a) > 0 and len(a[0]) > 0:
            a = a[0][0]
            fig, ax = plt.subplots()

            ax.imshow(image_array[idx], alpha=0.5)

            x = np.dot(conv_outputs[idx], class_weights[:,a])
                    
            ax.imshow(x, cmap='jet', alpha=0.5)
            ax.axis('off')

            fig.savefig(out_dir + '/' + str(index) + '_true_' + str(t_label).lower() + '_pred_' + str(p_label).lower() + ".jpg")
            plt.close()
    
    
def save_imgs(image_array, indices, true_labels, pred_labels, encoder, out_dir):
    image_array = np.array(image_array*255.0, dtype=np.int8)
    
    for idx in range(len(indices)):
        arr = image_array[idx]
        t_label = encoder.inverse_transform(true_labels[idx:idx+1])[0]
        p_label = encoder.inverse_transform(pred_labels[idx:idx+1])[0]
        index = indices[idx]
        
        img = Image.fromarray(arr, 'RGB')
        img.save(out_dir + '/' + str(index) + '_true_' + str(t_label).lower() + '_pred_' + str(p_label).lower() + ".jpg")
        
    
def predict(type='pt'):
    if type == 'pt':
        model = init_pt_model()
        model.load_weights('data/pt_model.h5')
        encoder = load_data_pkl('pt_encoder.pkl')
        pred_out_dir = 'product_type_predictions'
        cam_dir = 'product_type_cams'
        
    else:
        model = init_color_model()
        model.load_weights('data/color_model.h5')
        encoder = load_data_pkl('color_encoder.pkl')
        pred_out_dir = 'color_predictions'
        cam_dir = 'color_cams'
    
    test_indices = load_data_pkl('test_indices.pkl')
    n = 500 #len(test_indices)
    
    data_generator, test_labels, pred_labels = get_data_as_generator(n, batch_size, 'test', type), [], []
    
    total_batches = int(math.ceil(float(n)/batch_size))

    num_batches, start = 0, 0
    for batch_data, batch_labels in data_generator:
        test_labels += batch_labels.tolist()
        
        predictions = model.predict(batch_data)
        predictions = np.rint(predictions).astype(int).tolist()
        
        pred_labels += predictions
        num_batches += 1
        
        indices = [start + i for i in range(len(batch_labels))]
        
        save_imgs(batch_data, indices, np.array(batch_labels), np.array(predictions), encoder, pred_out_dir)
        cam(model, batch_data, indices, np.array(batch_labels), np.array(predictions), encoder, cam_dir)
        
        start += len(batch_labels)
        
        if num_batches == total_batches:
            break
            
    t_labels = encoder.inverse_transform(np.array(test_labels))
    p_labels = encoder.inverse_transform(np.array(pred_labels))
    
    print(classification_report(t_labels, p_labels))
            
read_input_file()
create_image_data()
create_data_pairs()
train_model()
predict(type='pt')
predict(type='color')
