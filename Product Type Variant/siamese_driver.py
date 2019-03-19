from siamese_network import SiameseNet
from siamese_api import SiameseAPI
import data_generator as dg
import pandas as pd, random, tables, numpy as np, os
import grouping_utils as gutils
import constants as cnt

print("Saving product data...")
gutils.save_product_data()
gutils.save_unique_items()

print("Getting items...")
items = gutils.load_data_pkl('items.pkl')

print("Creating tokens...")
dg.create_sent_tokens_array()

print("Training word2vec...")
gutils.train_wv_model()
gutils.generate_text_wv_embeddings()

print("Getting groups...")
groups = gutils.abstract_groups(items)
print(len(groups))

print("Getting wv embeddings one per group...")
random.seed(42)
group_indices = []
for abs_id, indexes in groups.items():
    idx = random.sample(indexes, 1)[0]
    group_indices.append(idx)

gutils.save_data_pkl(group_indices, cnt.GROUP_INDICES_FILE)

vectors = gutils.get_wv_embeddings(group_indices)

print("Constructing word embeddings KD Tree...")
gutils.construct_kd_tree(vectors, save_file=cnt.WV_KD_TREE_FILE)

group_indices = gutils.load_data_pkl(cnt.GROUP_INDICES_FILE)
print(items[group_indices[0]][0], items[group_indices[0]][5])
print(gutils.get_item_text(items[group_indices[0]]))

kdtree = gutils.load_data_pkl(cnt.WV_KD_TREE_FILE)
query_vector = gutils.get_wv_embeddings([group_indices[0]])[0]
u = gutils.get_nearest_neighbors_count(kdtree, query_vector, count=5)

for x in u:
    print(items[group_indices[x]][0], items[group_indices[x]][5])
    print(gutils.get_item_text(items[group_indices[x]]))

print("Generating data...")
num_train, num_test, num_validation = dg.generate_data(test_pct=0.2, validation_pct=0.2)
print(num_train, num_test, num_validation)

del(items)

print("Training Siamese...")
sapi = SiameseAPI()
sapi.train_model()

print(sapi.get_distance_threshold(threshold=0.95))

print("Inserting embeddings...")
sapi.insert_embeddings_pytables()

print("Constructing KD-Tree...")
vectors = sapi.fetch_embeddings_pytables()
gutils.construct_kd_tree(vectors, save_file=cnt.SIAMESE_KD_TREE_FILE)

print("Benchmarking suggested group API...")
print(sapi.benchmark_kdtree(num_samples=1000))