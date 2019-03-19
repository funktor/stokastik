import hashlib, itertools, json, pickle, cv2, math
from urllib.parse import urlparse, urlunparse, urljoin
import pandas as pd
import collections, os, random, numpy as np, glob, random
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.model_selection import train_test_split

def get_base_url(url):
    parsed_url = urlparse(url)
    return urlunparse([parsed_url[0], parsed_url[1], parsed_url[2], '', '', ''])

def url_to_filename(url):
    b_url = get_base_url(url)
    md5_hash = hashlib.md5(b_url.encode('utf-8')).hexdigest()
    return md5_hash + ".jpg"

def generate_synthetic_single_thread(img_path, image_save_dir, num_generate_per_instance, datagen):
    img_path_hash = hashlib.md5(img_path.encode('utf-8')).hexdigest()

    img = load_img(img_path)
    img_arr = img_to_array(img)
    img_arr = img_arr.reshape((1,) + img_arr.shape)

    image_prefix = img_path_hash

    counter = 0
    for batch in datagen.flow(img_arr, save_to_dir=image_save_dir, save_prefix=image_prefix, save_format='jpeg'):
        if counter >= 0.5*num_generate_per_instance:
            break
        counter += 1

    for size in random.sample(range(10, 1000), int(0.5*num_generate_per_instance)):
        img2 = img.resize((size, size), Image.ANTIALIAS)
        img2.save(os.path.join(image_save_dir, img_path_hash + '_resize_' + str(size) + '.jpg'))
        

def generate_synthetic_images(existing_images_path, image_save_dir, num_generate_per_instance=100, num_threads=5):
    image_files = glob.glob(os.path.join(existing_images_path, "*.*"))

    datagen = ImageDataGenerator(brightness_range=(0.1,1.2), 
                                 rotation_range=20,
                                 width_shift_range=0.15, 
                                 height_shift_range=0.15, 
                                 zoom_range=0.1, 
                                 horizontal_flip = True,
                                 vertical_flip = True,
                                 fill_mode='nearest')
    
    pool = ThreadPool(num_threads)
    http_codes = pool.map(lambda img_path: generate_synthetic_single_thread(img_path, image_save_dir, num_generate_per_instance, datagen), image_files)
    pool.close()
    pool.join()


def get_image_array_data(images_path, num_samples=3000, resized_image_size=64):
    images = glob.glob(os.path.join(images_path, "*.*"))
    images = random.sample(images, min(num_samples, len(images)))
    
    return np.array([img_to_array(load_img(image).resize((resized_image_size, resized_image_size))) for image in images])/255.0


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
    

def get_unimodel_data(placeholder_images_path, non_placeholder_images_path, prefix='train', num_samples_per_label=3000, image_size=64):
    placeholder_images_data = get_image_array_data(placeholder_images_path, num_samples_per_label, image_size)
    non_placeholder_images_data = get_image_array_data(non_placeholder_images_path, num_samples_per_label, image_size)
    
    labels = np.array([1]*placeholder_images_data.shape[0] + [0]*non_placeholder_images_data.shape[0])
    image_data = np.vstack((placeholder_images_data, non_placeholder_images_data))
    
    save_data_npy(labels, prefix + '_unimodel_labels.npy')
    save_data_npy(image_data, prefix + '_unimodel_image_data.npy')


def get_siamese_data(placeholder_images_path, non_placeholder_images_path, prefix='train', num_samples_per_label=3000, image_size=64):
    df = pd.read_csv('non_placeholder_image_urls.csv', sep=',')
    df = df.loc[df['urls'].notnull()]

    product_type, urls, all_pt_urls = list(df['product_type']), [json.loads(url) for url in list(df['urls'])], []

    product_type_url_map = collections.defaultdict(list)

    for i in range(len(product_type)):
        url = urls[i]
        if len(url) > 1:
            filename = url_to_filename(url[1])
        else:
            filename = url_to_filename(url[0])

        if os.path.exists(os.path.join(non_placeholder_images_path, filename)):
            product_type_url_map[product_type[i]].append(os.path.join(non_placeholder_images_path, filename))
            all_pt_urls += [os.path.join(non_placeholder_images_path, filename)]
        
    product_types = [key for key in product_type_url_map]
    all_pt_urls = set(all_pt_urls)
    
    random.seed(1)
    
    placeholders = glob.glob(os.path.join(placeholder_images_path, "*.*"))
    
    random.shuffle(placeholders)
    
    pos_data_pairs, neg_data_pairs = [], []
    url_hash_map, image_data = dict(), []
            
    for pd_type in product_types:
        curr_urls = product_type_url_map[pd_type]
        neg_urls = all_pt_urls.difference(set(curr_urls))
        
        random.shuffle(curr_urls)
        
        m = len(curr_urls)
        h = min(m-1, 15)

        for j in range(len(curr_urls)):
            for k in range(1, h+1):
                pos_data_pairs.append((curr_urls[j], curr_urls[(j+k)%m], 1, pd_type))
                
        if prefix == 'train':
            for j in range(len(curr_urls)):
                for negative_url in neg_urls:
                    neg_data_pairs.append((curr_urls[j], negative_url, 0, pd_type))
                
        for j in range(len(curr_urls)):
            for negative_url in placeholders:
                neg_data_pairs.append((curr_urls[j], negative_url, 0, pd_type))
            
    pos_data_pairs = random.sample(pos_data_pairs, min(num_samples_per_label, len(pos_data_pairs)))
            
    neg_data_pairs = random.sample(neg_data_pairs, min(num_samples_per_label, len(neg_data_pairs)))
    
    data_pairs = pos_data_pairs + neg_data_pairs
    
    for url1, url2, label, pd_type in data_pairs:
        if url1 not in url_hash_map:
            image_array = img_to_array(load_img(url1).resize((image_size,image_size)))
            image_data.append(image_array)
            url_hash_map[url1] = len(image_data)-1
            
        if url2 not in url_hash_map:
            image_array = img_to_array(load_img(url2).resize((image_size,image_size)))
            image_data.append(image_array)
            url_hash_map[url2] = len(image_data)-1
            
    if prefix == 'train':
        train_data_pairs, valid_data_pairs = train_test_split(data_pairs, test_size=0.2)
        
        save_data_npy(np.array(image_data), 'train_siamese_image_data.npy')
        save_data_pkl(url_hash_map, 'train_url_hash_map.pkl')
        save_data_pkl(train_data_pairs, 'train_data_pairs.pkl')
        save_data_pkl(valid_data_pairs, 'valid_data_pairs.pkl')
        
    else:
        save_data_npy(np.array(image_data), prefix + '_siamese_image_data.npy')
        save_data_pkl(url_hash_map, prefix + '_url_hash_map.pkl')
        save_data_pkl(data_pairs, prefix + '_data_pairs.pkl')
    

def get_image_data_unimodel(num_samples, prefix='train', batch_size=256):
    image_data = load_data_npy(prefix + "_unimodel_image_data.npy")
    labels = load_data_npy(prefix + "_unimodel_labels.npy")
    
    n = min(num_samples, image_data.shape[0])
    num_batches = int(math.ceil(float(n)/batch_size))
    
    np.random.seed(42)
    
    batch_num = 0
    
    while True:
        m = batch_num % num_batches
        
        if m == 0:
            p = np.random.permutation(n)
            image_data, labels = image_data[p], labels[p]
            
        start, end = m*batch_size, min((m+1)*batch_size, n)
        
        batch_num += 1
        
        yield image_data[start:end], labels[start:end]
    
def get_image_data_siamese(num_samples, prefix='train', batch_size=256):
    data_pairs = load_data_pkl(prefix + "_data_pairs.pkl")
    
    urls1, urls2, labels, pts = zip(*data_pairs)
    labels = np.array(labels)
    
    if prefix == 'valid':
        prefix = 'train'
        
    siamese_image_data = load_data_npy(prefix + "_siamese_image_data.npy")
    url_map = load_data_pkl(prefix + "_url_hash_map.pkl")
    
    url_indices1 = np.array([url_map[x] for x in urls1])
    url_indices2 = np.array([url_map[x] for x in urls2])
    
    n = min(num_samples, len(data_pairs))
    num_batches = int(math.ceil(float(n)/batch_size))
    
    np.random.seed(42)
    
    batch_num = 0
    
    while True:
        m = batch_num % num_batches
        
        if m == 0:
            p = np.random.permutation(n)
            url_indices1, url_indices2, labels = url_indices1[p], url_indices2[p], labels[p]
            
        start, end = m*batch_size, min((m+1)*batch_size, n)
    
        image_data_1 = siamese_image_data[url_indices1[start:end]]
        image_data_2 = siamese_image_data[url_indices2[start:end]]
        
        batch_num += 1
        
        yield [image_data_1, image_data_2], labels[start:end]