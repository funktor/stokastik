import importlib, random, numpy as np, collections
import data_generator as dg
importlib.reload(dg)
import siamese_predictor
importlib.reload(siamese_predictor)
from siamese_predictor import SiameseModel
from sklearn.metrics import classification_report

use_vgg = False
best_model = True

model = SiameseModel('data/siamese_model.h5', 
                     'data/siamese_current_best.h5', 
                     batch_size=256, 
                     training_samples=160000, 
                     validation_samples=40000, 
                     testing_samples=200000, 
                     use_vgg=use_vgg)

def test_model():
    model.init_model()
    model.load(best_model=best_model)
    model.score()
    
def ensemble_score(num_votes=11, frac_win=0.5, threshold=0.5, prefix='test'):
    model.init_model()
    model.load(best_model=best_model)
    
    train_data_pairs = dg.load_data_pkl("train_data_pairs.pkl")
    train_urls1, train_urls2, train_labels, train_pts = zip(*train_data_pairs)

    train_siamese_image_data = dg.load_data_npy("train_siamese_image_data.npy")

    train_url_map = dg.load_data_pkl("train_url_hash_map.pkl")
    
    pt_url_map = collections.defaultdict(list)
    
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            pt_url_map[train_pts[i]].append(train_urls1[i])

    test_data_pairs = dg.load_data_pkl(prefix + "_data_pairs.pkl")
    test_urls1, test_urls2, test_labels, test_pts = zip(*test_data_pairs)

    test_siamese_image_data = dg.load_data_npy(prefix + "_siamese_image_data.npy")

    test_url_map = dg.load_data_pkl(prefix + "_url_hash_map.pkl")
    
    visited, pred_labels, true_labels = set(), [], []
    
    for i in range(len(test_labels)):
        if test_labels[i] == 0:
            pt = test_pts[i]
            voting_urls = random.sample(pt_url_map[pt], min(num_votes, len(pt_url_map[pt])))
            voting_data = np.array([train_siamese_image_data[train_url_map[url]] for url in voting_urls])
            
            if test_urls2[i] not in visited:
                pl_data = test_siamese_image_data[test_url_map[test_urls2[i]]]
                pred_0 = model.score_ensemble(pl_data, voting_data, frac_win, threshold)
                pred_labels.append(pred_0)
                true_labels.append(0)
                visited.add(test_urls2[i])
                
            if test_urls1[i] not in visited:
                pr_data = test_siamese_image_data[test_url_map[test_urls1[i]]]
                pred_1 = model.score_ensemble(pr_data, voting_data, frac_win, threshold)
                pred_labels.append(pred_1)
                true_labels.append(1)
                visited.add(test_urls1[i])
    
    print(classification_report(true_labels, pred_labels))

def evaluate():
    model.fit()
    model.save()
    model.score()
    
evaluate()
test_model()
ensemble_score(num_votes=21, frac_win=0.5, threshold=0.75, prefix='test')
ensemble_score(num_votes=21, frac_win=0.5, threshold=0.75, prefix='test_irrelevant')
