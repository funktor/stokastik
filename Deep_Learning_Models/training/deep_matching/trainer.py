import sys, os
if "/data" in sys.path:
    sys.path.remove("/data")
sys.path.append("/home/jupyter/stormbreaker/deep_learning_models")
os.environ['BASE_PATH']="/home/jupyter/stormbreaker/deep_learning_models"
import numpy as np, pandas as pd, random
from sklearn.model_selection import train_test_split
import utilities.deep_matching.utilities as utils
import shared_utilities as shutils
from networks.deep_matching.network_tf import DeepMatchingNetwork
import stream_data_generators.deep_matching.generator as dg
import constants.deep_matching.constants as cnt

# print("Reading training file...")
# df_train = pd.read_csv(cnt.TRAIN_DATA_FILE_PATH, sep=",", encoding='utf-8')
# items_train = list(df_train.itertuples(index=False))

# print("Reading training file...")
# df_test = pd.read_csv(cnt.TEST_DATA_FILE_PATH, sep=",", encoding='utf-8')
# items_test = list(df_test.itertuples(index=False))

# if cnt.VECTOR_MODEL == 'fasttext':
#     print("Training fasttext for words...")
#     shutils.train_fasttext_model(items_train, utils.get_all_tokens_for_vector, cnt.FAST_TEXT_PATH_WORD, char_tokens=False)
    
#     print("Training fasttext for characters...")
#     shutils.train_fasttext_model(items_train, utils.get_all_tokens_for_vector, cnt.FAST_TEXT_PATH_CHAR, char_tokens=True)
    
# else:
#     print("Training word2vec for words...")
#     shutils.train_wv_model(items_train, utils.get_all_tokens_for_vector, cnt.WORD_VECT_PATH_WORD, char_tokens=False)
    
#     print("Training word2vec for characters...")
#     shutils.train_wv_model(items_train, utils.get_all_tokens_for_vector, cnt.WORD_VECT_PATH_CHAR, char_tokens=True)

# print("Getting train test tokens...")
# train_indices, test_indices = train_test_split(range(len(items_train)), test_size=0.2)

# shutils.save_data_pkl(train_indices, os.path.join(cnt.PERSISTENCE_PATH, "train_indices.pkl"))
# shutils.save_data_pkl(test_indices, os.path.join(cnt.PERSISTENCE_PATH, "test_indices.pkl"))

# train_data_pairs, test_data_pairs = utils.get_tokens_indices(items_train, train_indices), utils.get_tokens_indices(items_train, test_indices)

# train_data_pairs, test_data_pairs = utils.get_tokens_indices(items_train, range(len(items_train))), utils.get_tokens_indices(items_test, range(len(items_test)))

# shutils.save_data_pkl(train_data_pairs, os.path.join(cnt.PERSISTENCE_PATH, "train_data_pairs.pkl"))
# shutils.save_data_pkl(test_data_pairs, os.path.join(cnt.PERSISTENCE_PATH, "test_data_pairs.pkl"))

n = len(shutils.load_data_pkl(os.path.join(cnt.PERSISTENCE_PATH, "train_data_pairs.pkl")))
m = len(shutils.load_data_pkl(os.path.join(cnt.PERSISTENCE_PATH, "test_data_pairs.pkl")))

print("Training model...")
network = DeepMatchingNetwork(dg.get_data_as_generator, n, m)
network.fit()

# print("Scoring model...")
# network = DeepMatchingNetwork(dg.get_data_as_generator, n, m)
# network.scoring()

# network = DeepMatchingNetwork(dg.get_data_as_generator, n, m)
# network.init_model()
# network.load()


# test_indices = random.sample(test_indices, 500)

# vector_model = utils.get_vector_model(cnt.VECTOR_MODEL)
# out = []

# for i in range(len(test_indices)):
#     for j in range(len(test_indices)):
#         wm_title = str(items_train[i][3])
#         cm_title = str(items_train[j][4])

#         wm_tokens = shutils.padd_fn(shutils.get_tokens(wm_title), max_len=cnt.MAX_WORDS)
#         cm_tokens = shutils.padd_fn(shutils.get_tokens(cm_title), max_len=cnt.MAX_WORDS)
        
#         sent_data_1 = shutils.get_vectors(vector_model, [wm_tokens], cnt.VECTOR_DIM)
#         sent_data_2 = shutils.get_vectors(vector_model, [cm_tokens], cnt.VECTOR_DIM)
        
#         prob = network.predict_probability([sent_data_1, sent_data_2])[0][0]
        
#         if i == j:
#             out.append([wm_title, cm_title, prob, items_train[i][-1], 1])
#         else:
#             out.append([wm_title, cm_title, prob, 0, 0])
            
# df = pd.DataFrame(out, columns=["walmart_title", "amazon_title", "predicted_probability", "actual_match", "is_tagged"])
# df.to_csv("match_results.csv", sep=",")