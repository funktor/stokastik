import grouping_utils as gutils
import tables, collections, os
import numpy as np, math, random
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from gensim.models import Word2Vec
import constants as cnt

if os.path.exists(os.path.join(cnt.DATA_FOLDER, cnt.ITEMS_FILE)):
    items = gutils.load_data_pkl(cnt.ITEMS_FILE)
    groups = gutils.abstract_groups(items)

if os.path.exists(os.path.join(cnt.DATA_FOLDER, cnt.GROUP_INDICES_FILE)):
    group_indices = gutils.load_data_pkl(cnt.GROUP_INDICES_FILE)
    
if os.path.exists(os.path.join(cnt.DATA_FOLDER, cnt.WV_KD_TREE_FILE)):
    kdtree = gutils.load_data_pkl(cnt.WV_KD_TREE_FILE)
    
if os.path.exists(os.path.join(cnt.DATA_FOLDER, cnt.WV_MODEL_FILE)):
    wv_model = Word2Vec.load(os.path.join(cnt.DATA_FOLDER, cnt.WV_MODEL_FILE))

def create_sent_tokens_array():
    try:
        tokens_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SENT_TOKENS_FILE), mode='w')
        atom = tables.StringAtom(itemsize=16)
        tokens_arr = tokens_file.create_earray(tokens_file.root, 'data', atom, (0, cnt.MAX_WORDS))
        vocab = set()
        
        n, batch_size = len(items), cnt.PYTABLES_INSERT_BATCH_SIZE
        num_batches = int(math.ceil(float(n)/batch_size))

        for m in range(num_batches):
            start, end = m*batch_size, min((m+1)*batch_size, n)
            batch_items = [items[x] for x in range(start, end)]
            tokens = [gutils.padd_fn(gutils.get_tokens(gutils.get_item_text(item))) for item in batch_items]
            tokens_arr.append(tokens)
            vocab.update([x for token in tokens for x in token])
            
        vocab = sorted(list(vocab))
        word2idx_map = {w: i + 1 for i, w in enumerate(vocab)}
        gutils.save_data_pkl(word2idx_map, cnt.WORD2IDX_FILE)
        
        sent_tokens = tokens_file.root.data
        
        sents_arr_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SENT_ARRAYS_FILE), mode='w')
        atom = tables.Int32Atom()
        sents_arr = sents_arr_file.create_earray(sents_arr_file.root, 'data', atom, (0, cnt.MAX_WORDS))
        
        n, batch_size = len(items), cnt.PYTABLES_INSERT_BATCH_SIZE
        num_batches = int(math.ceil(float(n)/batch_size))
        
        for m in range(num_batches):
            start, end = m*batch_size, min((m+1)*batch_size, n)
            tokens = [sent_tokens[x] for x in range(start, end)]
            sent_arrs = [[gutils.word_to_idx(w, word2idx_map) for w in token] for token in tokens]
            sents_arr.append(sent_arrs)
        
    finally:
        tokens_file.close()
        sents_arr_file.close()
        
def generate_group_data(abs_id, next_abs_id, embeds, pt):
    curr_items = groups[abs_id]
    random.shuffle(curr_items)

    p = len(curr_items)
    
    embed_distances = np.array(euclidean_distances(embeds, embeds))
    max_distance = np.max(embed_distances)
    embed_distances = embed_distances.argsort(axis=None)[::-1]
    embed_distances = embed_distances[:min(len(embed_distances), cnt.NUM_PAIRS_PER_GROUP)]

    pos_data_pairs = [(curr_items[int(x/p)], curr_items[x%p], 1) for x in embed_distances]
    
    query_vector = embeds[0]

    dists, neg_items = gutils.get_nearest_neighbors_count(kdtree, query_vector, count=cnt.NUM_NEGATIVE_SAMPLES)
    neg_items = [neg_items[i] for i in range(len(neg_items)) if dists[i] > max_distance]
    neg_items = [group_indices[int(x)] for x in neg_items if math.isnan(x) is False]
    neg_items = [x for x in neg_items if items[x][5] != abs_id and items[x][1] == pt]

    if len(neg_items) == 0:
        neg_items = groups[next_abs_id]

    h = min(len(neg_items), len(curr_items), cnt.NUM_PAIRS_PER_GROUP)
    neg_items = random.sample(neg_items, h)
    
    neg_data_pairs = [(curr_items[x], neg_items[x], 0) for x in range(len(neg_items))]
    
    return pos_data_pairs, neg_data_pairs

def generate_data(test_pct=0.2, validation_pct=0.2):
    pt_abs_id_map = gutils.get_pt_abs_id_map(items)
    pos_data_pairs, neg_data_pairs = [], []

    for pt, abs_ids in pt_abs_id_map.items():
        selected_abs_ids = [x for x in abs_ids if len(groups[x]) >= cnt.MIN_GROUP_SIZE]
        n = len(selected_abs_ids)

        if n > 1:
            print(len(selected_abs_ids))
            random.shuffle(selected_abs_ids)
            group_embeds = {abs_id:gutils.get_wv_embeddings(groups[abs_id]) for abs_id in selected_abs_ids}
                
            pool = ThreadPool(cnt.NUM_THREADS)
            pt_data_pairs = pool.map(lambda x: generate_group_data(x[1], selected_abs_ids[(x[0]+1)%n], group_embeds[x[1]], pt), enumerate(selected_abs_ids))
            pool.close()
            pool.join()
            
            for x in pt_data_pairs:
                pos_data_pairs += x[0]
                neg_data_pairs += x[1]

    data_pairs = pos_data_pairs + neg_data_pairs

    train_data_pairs, test_data_pairs = train_test_split(data_pairs, test_size=test_pct)
    train_data_pairs, validation_data_pairs = train_test_split(train_data_pairs, test_size=validation_pct)

    gutils.save_data_pkl(train_data_pairs, cnt.TRAIN_DATA_PAIRS_FILE)
    gutils.save_data_pkl(test_data_pairs, cnt.TEST_DATA_PAIRS_FILE)
    gutils.save_data_pkl(validation_data_pairs, cnt.VALIDATION_DATA_PAIRS_FILE)

    return len(train_data_pairs), len(test_data_pairs), len(validation_data_pairs)
        
def get_data_as_generator(num_samples, prefix='train'):
    try:
        sent_tokens_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SENT_TOKENS_FILE), mode='r')
        sent_tokens = sent_tokens_file.root.data
        
        random.seed(42)

        data_pairs = gutils.load_data_pkl(prefix + "_data_pairs.pkl")
        random.shuffle(data_pairs)

        items1, items2, labels = zip(*data_pairs)
        items1, items2, labels = np.array(items1), np.array(items2), np.array(labels)

        n = len(data_pairs)
        num_batches = int(math.ceil(float(n)/cnt.SIAMESE_BATCH_SIZE))

        batch_num = 0

        while True:
            m = batch_num % num_batches

            start, end = m*cnt.SIAMESE_BATCH_SIZE, min((m+1)*cnt.SIAMESE_BATCH_SIZE, n)
            
            tokens1 = [sent_tokens[i] for i in items1[start:end]]
            tokens2 = [sent_tokens[i] for i in items2[start:end]]
            
            sent_data_1 = gutils.get_wv_siamese(wv_model, tokens1)
            sent_data_2 = gutils.get_wv_siamese(wv_model, tokens2)
            
            batch_num += 1

            yield [sent_data_1, sent_data_2], labels[start:end]
            
    finally:
        sent_tokens_file.close()
        

def get_data_as_vanilla(num_samples, prefix='train'):
    try:
        sent_tokens_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SENT_TOKENS_FILE), mode='r')
        sent_tokens = sent_tokens_file.root.data
        
        random.seed(42)
        
        data_pairs = gutils.load_data_pkl(prefix + "_data_pairs.pkl")
        random.shuffle(data_pairs)

        items1, items2, labels = zip(*data_pairs)
        items1, items2, labels = np.array(items1), np.array(items2), np.array(labels)

        n = min(num_samples, len(data_pairs))

        start, end = 0, n
        
        tokens1 = [sent_tokens[i] for i in items1[start:end]]
        tokens2 = [sent_tokens[i] for i in items2[start:end]]

        sent_data_1 = gutils.get_wv_siamese(wv_model, tokens1)
        sent_data_2 = gutils.get_wv_siamese(wv_model, tokens2)

        return [sent_data_1, sent_data_2], labels[start:end]
    
    finally:
        sent_tokens_file.close()
        
