import hashlib, itertools, json, pickle, cv2, math, io, time, sys
import collections, os, random, numpy as np, glob, random, itertools
import constants as cnt
from siamese_predictor import SiameseModel
from sklearn.neighbors import KDTree
from sklearn.metrics import classification_report
from collections import Counter
import utilities as utils
import json


def get_voting_embeddings(model):
    voting_data = utils.get_sampled_voting_data()
    total_samples = np.sum([len(x) for pt, x in voting_data.items()])
    
    voting_embeddings = np.empty((total_samples, cnt.EMBEDDING_SIZE))
    voting_pts = []
    
    start = 0
    for pt, v_data in voting_data.items():
        embeds = model.get_embeddings(v_data)
        voting_embeddings[start:start+v_data.shape[0]] = embeds
        voting_pts += [pt] * v_data.shape[0]
        start += v_data.shape[0]
    
    return voting_embeddings, np.array(voting_pts)


def compute_embeddings(model):
    voting_embeddings, voting_pts = get_voting_embeddings(model)
    voting_kd_tree = KDTree(voting_embeddings, leaf_size=cnt.KD_TREE_LEAF_SIZE)
    
    utils.save_data_pkl(voting_pts, cnt.VOTING_PTS_PATH)
    utils.save_data_pkl(voting_kd_tree, cnt.VOTING_KD_TREE_PATH)
    
    return voting_kd_tree, voting_pts
    
    
def get_prediction(test_embeddings, model, voting_kd_tree, voting_pts, distance_thres, slope):
    m = len(set(voting_pts))
    distance_thres, slope = model.get_distance_threshold()
    
    nearest = voting_kd_tree.query_radius(test_embeddings, r=distance_thres)
    
    if slope < 0:
        nearest = [list(set(range(len(voting_pts))).difference(set(x))) for x in nearest]
        
    pred_labels = [0]*test_embeddings.shape[0]
    confidences = []
    
    for i in range(len(nearest)):
        x = nearest[i]
        confidences.append(1.0-(len(x)/len(voting_pts)))
        
        if len(x) > 0:
            pred_labels[i] = 1 if Counter(voting_pts[x]).most_common(1)[0][1] < len(x)/m else 0
        else:
            pred_labels[i] = 1
    
    return pred_labels, confidences


def fetch_test_embeddings(image_data, model):
    n, batch_size = image_data.shape[0], cnt.EMBEDDING_BATCH_SIZE
    num_batches = int(math.ceil(float(n)/batch_size))
    embeddings = np.empty((n, cnt.EMBEDDING_SIZE))

    for m in range(num_batches):
        start, end = m*batch_size, min((m+1)*batch_size, n)
        embeddings[start:end] = model.get_embeddings(image_data[start:end])
        
    return embeddings
    
    
def ensemble_score(model, voting_kd_tree, voting_pts, distance_thres, slope):
    test_image_data = utils.load_data_npy(cnt.TEST_IMAGE_DATA_FILE)
    pr_embeddings = fetch_test_embeddings(test_image_data, model)
    
    placeholder_data = utils.load_data_npy(cnt.TEST_PLACEHOLDERS_FILE)
    pl_embeddings = fetch_test_embeddings(placeholder_data, model)
    
    all_embeddings = np.vstack((pr_embeddings, pl_embeddings))
    
    pred_labels, confidences = get_prediction(all_embeddings, model, voting_kd_tree, voting_pts, distance_thres, slope)
    true_labels = [0]*pr_embeddings.shape[0] + [1]*pl_embeddings.shape[0]
    
    print(classification_report(true_labels, pred_labels))
    

class SiameseService(object):
    def __init__(self):
        self.siamese_model = SiameseModel()
        self.siamese_model.init_model()
        
        if os.path.exists(os.path.join(cnt.DATA_DIR, cnt.SIAMESE_BEST_MODEL_PATH)) or os.path.exists(os.path.join(cnt.DATA_DIR, cnt.SIAMESE_MODEL_PATH)):
            self.siamese_model.load()
            self.distance_thres, self.slope = self.siamese_model.get_distance_threshold()
        else:
            self.distance_thres, self.slope = None, None
            
        if os.path.exists(os.path.join(cnt.DATA_DIR, cnt.VOTING_KD_TREE_PATH)):
            self.voting_kd_tree = utils.load_data_pkl(cnt.VOTING_KD_TREE_PATH)
        else:
            self.voting_kd_tree = None
            
        if os.path.exists(os.path.join(cnt.DATA_DIR, cnt.VOTING_PTS_PATH)):
            self.voting_pts = utils.load_data_pkl(cnt.VOTING_PTS_PATH)
        else:
            self.voting_pts = None
        
        
    def train(self):
        self.siamese_model.fit()
        self.voting_kd_tree, self.voting_pts = compute_embeddings(self.siamese_model)
        self.distance_thres, self.slope = self.siamese_model.get_distance_threshold()
        
    
    def evaluate(self):
        self.load()
            
        print("Test data score")
        self.siamese_model.score()
        
        print("Test data score with voting")
        ensemble_score(self.siamese_model, self.voting_kd_tree, self.voting_pts, self.distance_thres, self.slope)
        
    
    def save(self):
        self.siamese_model.save()
        
        utils.save_data_pkl(self.voting_pts, cnt.VOTING_PTS_PATH)
        utils.save_data_pkl(self.voting_kd_tree, cnt.VOTING_KD_TREE_PATH)
        
    
    def load(self):
        if self.siamese_model.model is None or self.voting_kd_tree is None or self.voting_pts is None or self.distance_thres is None:
            self.siamese_model.init_model()
            self.siamese_model.load()

            self.voting_kd_tree = utils.load_data_pkl(cnt.VOTING_KD_TREE_PATH)
            self.voting_pts = utils.load_data_pkl(cnt.VOTING_PTS_PATH)

            self.distance_thres, self.slope = self.siamese_model.get_distance_threshold()
        
        
    def predict(self, urls, img_arrs=None):
        try:
            self.load()

            if img_arrs is None:
                img_arrs = [utils.image_url_to_array(url) for url in urls]
            
            failed_output = [{'index':i, 'url':urls[i], 'status':'failure', 'message':'Image could not be retrieved', 'label':-1, 'confidence':-1} for i in range(len(urls)) if isinstance(img_arrs[i], int)]
            
            non_failed_img_indices = [i for i in range(len(urls)) if isinstance(img_arrs[i], int) is False]
            correct_img_arrs = [img_arrs[i] for i in non_failed_img_indices]
            
            embeddings = fetch_test_embeddings(np.array(correct_img_arrs), self.siamese_model)
            pred_labels, confidences = get_prediction(embeddings, self.siamese_model, self.voting_kd_tree, self.voting_pts, self.distance_thres, self.slope)
            
            valid_output = []
            
            j = 0
            for i in non_failed_img_indices:
                if pred_labels[j] == 1:
                    valid_output.append({'index':i, 'url':urls[i], 'status':'success', 'tag':'placeholder', 'label':1, 'confidence':confidences[j]})
                else:
                    valid_output.append({'index':i, 'url':urls[i], 'status':'success', 'tag':'non_placeholder', 'label':0, 'confidence':1-confidences[j]})
                j += 1
                        
            return json.dumps({'output':failed_output + valid_output, 'status':'completed'})
            
        except Exception as err:
            return json.dumps({'status':'failure', 'message':err.message})