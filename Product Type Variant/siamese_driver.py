from siamese_network import SiameseNet
from siamese_api import SiameseAPI
import data_generator as dg
import pandas as pd, random, tables, numpy as np, os, math, importlib, time
import grouping_utils as gutils
import constants as cnt

# print("Saving product data...")
# gutils.save_product_data()
# gutils.save_unique_items()

# print("Getting items...")
# items = gutils.load_data_pkl('items.pkl')

# print("Creating tokens...")
# importlib.reload(dg)
# dg.create_sent_tokens_array()

# print("Training word2vec...")
# gutils.train_wv_model()
# gutils.generate_text_wv_embeddings()

# print("Getting groups...")
# groups = gutils.abstract_groups(items)
# print(len(groups))

# print("Getting wv embeddings one per group...")
# random.seed(42)
# group_indices = []
# for abs_id, indexes in groups.items():
#     idx = random.sample(indexes, 1)[0]
#     group_indices.append(idx)

# gutils.save_data_pkl(group_indices, cnt.GROUP_INDICES_FILE)

# vectors = gutils.get_wv_embeddings(group_indices)

# print("Constructing word embeddings KD Tree...")
# gutils.construct_kd_tree(vectors, save_file=cnt.WV_KD_TREE_FILE)

# group_indices = gutils.load_data_pkl(cnt.GROUP_INDICES_FILE)
# print(items[group_indices[0]][0], items[group_indices[0]][5])
# print(gutils.get_item_text(items[group_indices[0]]))

# kdtree = gutils.load_data_pkl(cnt.WV_KD_TREE_FILE)
# query_vector = gutils.get_wv_embeddings([group_indices[0]])[0]
# u = gutils.get_nearest_neighbors_count(kdtree, query_vector, count=5)

# for x in u[1]:
#     print(items[group_indices[x]][0], items[group_indices[x]][5])
#     print(gutils.get_item_text(items[group_indices[x]]))

# print("Generating data...")
# importlib.reload(dg)
# num_train, num_test, num_validation = dg.generate_data(test_pct=0.2, validation_pct=0.2)
# print(num_train, num_test, num_validation)

# del(items)

# print("Training Siamese...")
# sapi = SiameseAPI()
# sapi.train_model()

# print(sapi.get_distance_threshold(threshold=0.95))

# print("Inserting embeddings...")
# sapi.insert_embeddings_pytables()

# print("Constructing KD-Tree...")
# vectors = sapi.fetch_embeddings_pytables()
# gutils.construct_kd_tree(vectors, save_file=cnt.SIAMESE_KD_TREE_FILE)

# print("Benchmarking suggested group API...")
# print(gutils.benchmark_kdtree(num_samples=1000))

# print("Getting all items...")
# items_4200 = gutils.load_data_pkl('items.pkl')
# items_13 = gutils.load_data_pkl('items_13.pkl')

# all_items = items_4200 + items_13
# gutils.save_data_pkl(all_items, 'all_items.pkl')

    
print("Computing embeddings...")
items_13 = gutils.load_data_pkl('items_13.pkl')
print(len(items_13))

# try:
#     sapi = SiameseAPI()
    
#     sapi.get_model()
#     sapi.model.init_model()
#     sapi.model.load()

#     embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, 'embeddings_13.h5'), mode='w')
#     atom = tables.Float32Atom()
#     embeds_arr = embeds_file.create_earray(embeds_file.root, 'data', atom, (0, cnt.SIAMESE_EMBEDDING_SIZE))

#     n, batch_size = len(items_13), cnt.PYTABLES_INSERT_BATCH_SIZE
#     num_batches = int(math.ceil(float(n)/batch_size))

#     for m in range(num_batches):
#         start, end = m*batch_size, min((m+1)*batch_size, n)
#         batch_items = [items_13[x] for x in range(start, end)]
#         batch_sentences = [gutils.get_item_text(item) for item in batch_items]
#         batch_embeds = sapi.get_representations(batch_sentences)
#         embeds_arr.append(batch_embeds)

# finally:
#     embeds_file.close()
    
# print("KD Tree...")   
# sapi = SiameseAPI()
# vectors_4200 = sapi.fetch_embeddings_pytables()
    
try:
    embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, 'embeddings_13.h5'), mode='r')
    embeds_arr = embeds_file.root.data
    vectors_13 = np.array(embeds_arr[:,:])
finally:
    embeds_file.close()
    
# print(vectors_13.shape)
    
# vectors = np.vstack((vectors_4200, vectors_13))
    
# try:
#     embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, 'embeddings_all.h5'), mode='w')
#     atom = tables.Float32Atom()
#     embeds_arr = embeds_file.create_earray(embeds_file.root, 'data', atom, (0, cnt.SIAMESE_EMBEDDING_SIZE))
    
#     embeds_arr.append(vectors)
# finally:
#     embeds_file.close()
    
pts = [item[1] for item in items_13]

# start = time.time()
# gutils.construct_kd_tree_per_PT(vectors_13, pts, save_file='kd_tree_new_13_pt.pkl')
# print(time.time()-start)

print("Quantizing...")
start = time.time()
gutils.construct_product_quantizer_per_PT(vectors_13, pts, save_file='pq.pkl')
print(time.time()-start)