import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from PIL import Image
import collections, tables, pandas as pd, os, pickle
from keras.preprocessing.image import img_to_array, load_img
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy, json
import os, random, numpy as np, math
import constants.attribute_extraction.constants as cnt
import shared_utilities as shutils
from gensim.models import Word2Vec, FastText
from collections import defaultdict

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

def read_input_file():
    df = pd.read_csv(cnt.INPUT_FILE_PATH, sep='\\t', engine='python')
    df = df.dropna()
    df = df.groupby('age_group').filter(lambda x : len(x)>=200)
    df.drop_duplicates(subset=['item_id'], inplace=True)
    df = df.apply(lambda row: url_type_fn(row), axis=1)
    urls = list(df.image_urls)
    filenames = [os.path.join(cnt.DOWNLOADED_IMAGES_PATH, shutils.url_to_filename(url)) for url in urls]
    df['image_path'] = filenames
    
    with open(cnt.URLS_LIST_PATH, 'w') as f:
        for url in urls:
            f.write(url+'\n')
    f.close()
    df.to_csv(cnt.OUTPUT_FILE_PATH, sep=",", encoding='utf-8')
    
    
def download_images():
    if not os.path.exists(cnt.DOWNLOADED_IMAGES_PATH):
        os.mkdir(cnt.DOWNLOADED_IMAGES_PATH)
        
    shutils.fetch_urls_parallel(cnt.URLS_LIST_PATH, cnt.DOWNLOADED_IMAGES_PATH, 6)
    
    
def create_image_data():
    try:
        img_arr_file = tables.open_file(cnt.IMAGE_ARRAY_PATH, mode='w')
        atom = tables.Float32Atom()
        img_arr = img_arr_file.create_earray(img_arr_file.root, 'data', atom, (0, cnt.IMAGE_SIZE, cnt.IMAGE_SIZE, 3))

        chunk_size, labels = 5000, []
        for df_chunk in pd.read_csv(cnt.OUTPUT_FILE_PATH, chunksize=chunk_size):
            df = df_chunk[list(df_chunk['image_path'].apply(lambda x:os.path.exists(x)))]
            print(df.shape)
            labels += list(df['age_group'])
            file_paths = list(df['image_path'])
            img_arr.append([img_to_array(load_img(image).convert('RGB').resize((cnt.IMAGE_SIZE, cnt.IMAGE_SIZE)))/255.0 for image in file_paths])
            
        shutils.save_data_pkl(labels, cnt.LABELS_PATH)
    finally:
        img_arr_file.close()
        
        
def get_all_tokens_for_vector(items, char_tokens=False):
    text_data = [str(item[0]) for item in items]
    all_tokens = [shutils.padd_fn(shutils.get_tokens(x, char_tokens=char_tokens), max_len=cnt.MAX_WORDS) for x in text_data]
    
    return all_tokens

def get_vector_model(vector_model_id='wv', char_tokens=False):
    if vector_model_id == 'fasttext':
        if char_tokens:
            return FastText.load(cnt.FAST_TEXT_PATH_CHAR)
        return FastText.load(cnt.FAST_TEXT_PATH_WORD)
    else:
        if char_tokens:
            return Word2Vec.load(cnt.WORD_VECT_PATH_CHAR)
        return Word2Vec.load(cnt.WORD_VECT_PATH_WORD)
    
        
def create_text_data():
    df = pd.read_csv(cnt.OUTPUT_FILE_PATH)
    df = df[list(df['image_path'].apply(lambda x:os.path.exists(x)))]

    titles = list(df['title'])
    short_desc = list(df['short_description'])
    long_desc = list(df['long_description'])

    class_labels = [x.strip().split('__') for x in list(df['age_group'])]

    text_corpus = [titles[i] + " " + short_desc[i] + " " + long_desc[i] for i in range(len(titles))]

    features = set(shutils.get_features_mi(text_corpus, class_labels, cnt.MAX_FEATURES))
    input_tensor, nn_tokenizer = shutils.get_preprocessed_data(text_corpus, feature_set=features, max_length=cnt.MAX_WORDS)
    
    shutils.save_data_pkl(input_tensor, cnt.INPUT_TENSOR_PATH)
    shutils.save_data_pkl(nn_tokenizer, cnt.TENSOR_TOKENIZER_PATH)
    
    vocab_size = len(nn_tokenizer.word_index)+1
    shutils.save_data_pkl(vocab_size, cnt.VOCAB_SIZE_PATH)
        
    
def create_train_test():
    try:
        img_arr_file = tables.open_file(cnt.IMAGE_ARRAY_PATH, mode='r')
        img_arr = img_arr_file.root.data
        
        train_indices, test_indices = train_test_split(range(img_arr.shape[0]), test_size=0.15)
        
        encoder = LabelBinarizer()
        
        labels = shutils.load_data_pkl(cnt.LABELS_PATH)
        transfomed_labels = encoder.fit_transform(labels)
        
        shutils.save_data_pkl(transfomed_labels, cnt.TRANSFORMED_LABELS_PATH)
        shutils.save_data_pkl(encoder, cnt.ENCODER_PATH)
        
        print(len(train_indices), len(test_indices))
        
        shutils.save_data_pkl(train_indices, cnt.TRAIN_INDICES_PATH)
        shutils.save_data_pkl(test_indices, cnt.TEST_INDICES_PATH)
            
    finally:
        img_arr_file.close()

    
def cam(model, image_array, indices, true_labels, pred_labels, encoder, out_dir):
    image_array = np.array(image_array*255, dtype=np.uint8)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    class_weights = model.layers[-1].get_weights()[0]
    
    get_last_conv_output = K.function([model.layers[0].input, model.layers[1].get_input_at(0)], [model.layers[1].layers[-4].get_output_at(0)])
    
    conv_outputs = get_last_conv_output([image_array, image_array])[0]
    conv_outputs = scipy.ndimage.zoom(conv_outputs, (1, 8, 8, 1), order=1)
    
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
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for idx in range(len(indices)):
        arr = image_array[idx]
        t_label = encoder.inverse_transform(true_labels[idx:idx+1])[0]
        p_label = encoder.inverse_transform(pred_labels[idx:idx+1])[0]
        index = indices[idx]
        
        img = Image.fromarray(arr, 'RGB')
        img.save(out_dir + '/' + str(index) + '_true_' + str(t_label).lower() + '_pred_' + str(p_label).lower() + ".jpg")
        
        
def custom_classification_scores(true_labels_names, pred_labels_names):
    tp, fp, fn = defaultdict(float), defaultdict(float), defaultdict(float)
    support = defaultdict(float)

    for idx in range(len(true_labels_names)):
        true_label, pred_label = list(true_labels_names[idx]), list(pred_labels_names[idx])

        for label in pred_label:
            if label in true_label:
                tp[label] += 1
            else:
                fp[label] += 1

        for label in true_label:
            support[label] += 1

            if label not in pred_label:
                fn[label] += 1

    precision, recall, f1_score = defaultdict(float), defaultdict(float), defaultdict(float)

    tot_precision, tot_recall, tot_f1 = 0.0, 0.0, 0.0
    sum_sup = 0.0

    for label, sup in support.items():
        precision[label] = float(tp[label])/(tp[label] + fp[label]) if label in tp else 0.0
        recall[label] = float(tp[label])/(tp[label] + fn[label]) if label in tp else 0.0

        f1_score[label] = 2 * float(precision[label] * recall[label])/(precision[label] + recall[label]) if precision[label] + recall[label] != 0 else 0.0

        tot_f1 += sup * f1_score[label]
        tot_precision += sup * precision[label]
        tot_recall += sup * recall[label]

        sum_sup += sup

    for label, sup in support.items():
        print (label, precision[label], recall[label], f1_score[label], sup)

    return (tot_precision/float(sum_sup), tot_recall/float(sum_sup), tot_f1/float(sum_sup), sum_sup)

    
        