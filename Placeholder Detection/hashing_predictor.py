import imagehash, os, glob, itertools, numpy as np
from keras.preprocessing.image import img_to_array, load_img
import data_generator as dg
from sklearn.metrics import classification_report

class Hashing(object):
    def __init__(self, threshold=15):
        self.hashes = []
        self.threshold = threshold
        
    def fit(self, image_files):
        for img_file in image_files:
            img = load_img(img_file)
            self.hashes.append(imagehash.phash(img))
            
    def predict(self, image_files):
        predictions = []
        for img_file in image_files:
            img = load_img(img_file)
            im_hash = imagehash.phash(img)
            
            min_dist = float("Inf")
            for x in self.hashes:
                min_dist = min(min_dist, abs(im_hash-x))
                
            if min_dist < self.threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions
    
    def score(self, pos_image_files, neg_image_files):
        labels = [1]*len(pos_image_files) + [0]*len(neg_image_files)
        preds = self.predict(pos_image_files) + self.predict(neg_image_files)
        
        print classification_report(labels, preds)
        
    def save(self):
        dg.save_data_npy(np.array(self.hashes), "trained_phashes.npy")
        
    def load(self):
        self.hashes = list(dg.load_data_npy("trained_phashes.npy"))