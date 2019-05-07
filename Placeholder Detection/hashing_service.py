import imagehash, os, glob, itertools, numpy as np
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report
import constants as cnt
import utilities as utils
import json


def is_placeholder(img, hash_list):
    im_hash = imagehash.phash(img)
            
    min_dist = float("Inf")
    for x in hash_list:
        min_dist = min(min_dist, abs(im_hash-x))
        
    pred = 1 if min_dist < cnt.PHASH_MATCHING_THRESHOLD else 0
    conf = 1.0-(min_dist/64.0)

    return pred, conf


class HashingService(object):
    def __init__(self):
        if os.path.exists(os.path.join(cnt.DATA_DIR, cnt.HASH_LIST_PATH)):
            self.hash_list = utils.load_data_npy(cnt.HASH_LIST_PATH)
        else:
            self.hash_list = []
        
    def train(self):
        image_files = glob.glob(os.path.join(cnt.TAGGED_PLACEHOLDER_IMAGES_PATH, "*.*"))
        
        for img_file in image_files:
            img = load_img(img_file).resize((cnt.IMAGE_SIZE, cnt.IMAGE_SIZE))
            self.hash_list.append(imagehash.phash(img))
    
    def evaluate(self):
        self.load()
        
        product_type_dirs = glob.glob(os.path.join(cnt.PRODUCT_IMAGES_PATH, "*/"))
        product_image_files = []
        for pt_dir in product_type_dirs:
            product_image_files += glob.glob(os.path.join(pt_dir, "*.*"))
            
        placeho_image_files = glob.glob(os.path.join(cnt.TEST_PLACEHOLDER_IMAGES_PATH, "*.*"))
        
        image_files = product_image_files + placeho_image_files
        true_labels = [0] * len(product_image_files) + [1] * len(placeho_image_files)
        
        pred_labels = [is_placeholder(load_img(img_file).resize((cnt.IMAGE_SIZE, cnt.IMAGE_SIZE)), self.hash_list)[0] for img_file in image_files]
        
        print(classification_report(true_labels, pred_labels))
    
    def save(self):
        utils.save_data_npy(np.array(self.hash_list), cnt.HASH_LIST_PATH)
    
    def load(self):
        if len(self.hash_list) == 0:
            self.hash_list = utils.load_data_npy(cnt.HASH_LIST_PATH)
        
    def predict(self, urls, imgs=None):
        try:
            self.load()

            if imgs is None:
                imgs = [utils.image_url_to_obj(url) for url in urls]

            output = []

            for i in range(len(urls)):
                img = imgs[i]

                if isinstance(img, int):
                    output.append({'index':i, 'url':urls[i], 'status':'failure', 'message':'Image could not be retrieved', 'label':-1, 'confidence':-1})

                prediction, confidence = is_placeholder(img, self.hash_list)

                if prediction == 1:
                    output.append({'index':i, 'url':urls[i], 'status':'success', 'tag':'placeholder', 'label':1, 'confidence':confidence})
                else:
                    output.append({'index':i, 'url':urls[i], 'status':'success', 'tag':'non_placeholder', 'label':0, 'confidence':1-confidence})

            return json.dumps({'output':output, 'status':'completed'})
        
        except Exception as err:
            return json.dumps({'status':'failure', 'message':err.message})
