import hashlib, itertools, json, pickle, cv2, math
import pandas as pd
import collections, os, random, numpy as np, glob, random, itertools
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
import constants as cnt
import utilities as utils

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
    
    
def get_image_data_siamese(num_samples, prefix='train'):
    random.seed(42)
    
    batch_size = cnt.SIAMESE_BATCH_SIZE
    train_image_data, train_pt_url_map = utils.get_sampled_train_data()
    
    negative_samples_pts = dict()
    for pt, urls in train_pt_url_map.items():
        w = len(urls)
        negative_samples_pts[pt] = list(set(range(train_image_data.shape[0])).difference(set(urls)))
    
    image_data = utils.load_data_npy(prefix + "_image_data.npy")
    p = np.random.permutation(image_data.shape[0])
    
    pt_url_map = utils.load_data_pkl(prefix + "_pt_url_map.pkl")
    url_pt_map = utils.load_data_pkl(prefix + "_url_pt_map.pkl")
    
    test_placeholder_data = utils.load_data_npy(cnt.TEST_PLACEHOLDERS_FILE)
    train_placeholder_data = utils.load_data_npy(cnt.TRAIN_PLACEHOLDERS_FILE)
    
    random.seed(None)
    
    half_batch_size = int(0.5*batch_size)
    
    n = image_data.shape[0]
    num_batches = int(math.ceil(float(n)/half_batch_size))
    
    batch_num = 0
    
    while True:
        m = batch_num % num_batches
        start, end = m*half_batch_size, min((m+1)*half_batch_size, n)
        d = batch_size-half_batch_size
        
        a1_samples = p[start:end]
        a1 = image_data[a1_samples]
        a1_pts = [url_pt_map[x] for x in a1_samples]
        
        b1_samples = [random.sample(train_pt_url_map[pt], 1)[0] for pt in a1_pts]
        b1 = train_image_data[b1_samples]
        
        a2_samples = np.random.choice(range(image_data.shape[0]), d, replace=False)
        a2 = image_data[a2_samples]
        a2_pts = [url_pt_map[x] for x in a2_samples]
        
        b2_samples = [random.sample(negative_samples_pts[pt], 1)[0] for pt in a2_pts]
        b2 = train_image_data[b2_samples]
        
#         if prefix == 'train':
#             b3_samples = np.random.choice(range(train_placeholder_data.shape[0]), d, replace=True)
#             b3 = train_placeholder_data[b3_samples]
            
#         else:
        b3_samples = np.random.choice(range(test_placeholder_data.shape[0]), d, replace=True)
        b3 = test_placeholder_data[b3_samples]
        
        b2 = np.vstack((b2, b3))
        b2 = b2[np.random.randint(b2.shape[0], size=d), :]
        
        image_data_1 = np.vstack((a1, a2))
        image_data_2 = np.vstack((b1, b2))
        
        labels = np.array([0] * b1.shape[0] + [1] * b2.shape[0])
        
        q = np.random.permutation(image_data_1.shape[0])
        image_data_1, image_data_2, labels = image_data_1[q], image_data_2[q], labels[q]
        
        batch_num += 1
        
        yield [image_data_1, image_data_2], labels