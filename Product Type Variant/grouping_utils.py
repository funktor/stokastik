import pickle, os, re, numpy as np, gensim, time
import pandas as pd, math, collections, random, tables
from sklearn.model_selection import train_test_split
from multiprocessing.dummy import Pool as ThreadPool
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.neighbors import KDTree
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import constants as cnt

def save_product_data():
    df_chunk = pd.read_csv('data/pcf_dump.tsv', sep='\\t', engine='python', encoding='utf-8')
    
    product_data = list(zip(list(df_chunk['item_id']), 
                            list(df_chunk['product_type']), 
                            list(df_chunk['title']), 
                            list(df_chunk['long_description']), 
                            list(df_chunk['short_description']), 
                            list(df_chunk['abstract_product_id']), 
                            list(df_chunk['brand']), 
                            list(df_chunk['size']), 
                            list(df_chunk['color'])))
    print(len(product_data))

    save_data_pkl(product_data, 'product_data.pkl')
    
def save_unique_items():
    product_data = load_data_pkl(cnt.PRODUCT_DATA_FILE)
    
    items = get_unique_items_pt(product_data)
    save_data_pkl(items, cnt.ITEMS_FILE)
    print(len(items))
    
def abstract_groups(items):
    clusters = defaultdict(list)

    for idx in range(len(items)):
        abs_pd_id = items[idx][5]
        clusters[abs_pd_id].append(idx)

    return clusters

def get_item_text(item):
    txt = str(item[2]) + " " + str(item[4])
    return txt.encode("ascii", errors="ignore").decode()

def word_to_idx(word, word2idx_map):
    return word2idx_map[word] if word in word2idx_map else 0

def padd_fn(x):
    return x + ['PAD_TXT']*(cnt.MAX_WORDS - len(x)) if len(x) < cnt.MAX_WORDS else x[:cnt.MAX_WORDS]

def clean_tokens(tokens, to_replace='[^\w\-\+\&\.\'\"\:]+'):
    lemma = WordNetLemmatizer()
    
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    tokens = [lemma.lemmatize(token) for token in tokens]
    
    return tokens

def tokenize(mystr):
    tokenizer = RegexpTokenizer('[^ ]+')
    mystr = mystr.lower()

    return tokenizer.tokenize(mystr)

def get_tokens(sentence, to_replace='[^\w\-\+\&\.\'\"\:]+'):
    sentence = re.sub('<[^<]+?>', ' ', str(sentence))
    sentence = re.sub(to_replace, ' ', str(sentence))
    
    tokens = clean_tokens(tokenize(sentence), to_replace)
    tokens = [token.strip() for token in tokens]
    
    return tokens

def get_unique_items_pt(product_data, pt=None):
    if pt is not None:
        items = [x for x in product_data if x[1] == pt]
    else:
        items = product_data
    
    unique_items, added_item_ids = [], set()

    for item in items:
        if item[0] not in added_item_ids:
            unique_items.append(item)
            added_item_ids.add(item[0])
    
    return unique_items

def save_data_npy(data, path):
    np.save(os.path.join("data", path), data)
    
def load_data_npy(path):
    return np.load(os.path.join("data", path))
    
def save_data_pkl(data, path):
    with open(os.path.join("data", path), 'wb') as f:
        pickle.dump(data, f, protocol=4)
        
def load_data_pkl(path):
    with open(os.path.join("data", path), 'rb') as f:
        return pickle.load(f)

def construct_kd_tree(vectors, save_file=cnt.SIAMESE_KD_TREE_FILE):
    kdtree = KDTree(vectors, leaf_size=cnt.KD_TREE_LEAF_SIZE)
    save_data_pkl(kdtree, save_file)
    
def get_nearest_neighbors_radius(kdtree, query_vector, query_radius):
    nearest = kdtree.query_radius([query_vector], r=query_radius)
    return nearest[0]

def get_nearest_neighbors_count(kdtree, query_vector, count):
    dist, nearest = kdtree.query([query_vector], k=count)
    return dist[0], nearest[0]

def benchmark_kdtree(kdtree, query_radius, num_samples=10000):
    items = load_data_pkl(cnt.ITEMS_FILE)
    sentences = [item[2] for item in random.sample(items, min(num_samples, len(items)))]

    times = []
    for sentence in sentences:
        start = time.time()
        get_nearest_neighbors_radius(kdtree, sentence, query_radius)
        times.append(time.time()-start)

    return np.mean(times), np.median(times), np.max(times)

def lcs(tokens1, tokens2):
    cache = collections.defaultdict(dict)
    for i in range(-1, len(tokens1)):
        for j in range(-1, len(tokens2)):
            if i == -1 or j == -1:
                cache[i][j] = 0
            else:
                if tokens1[i] == tokens2[j]:
                    cache[i][j] = cache[i - 1][j - 1] + 1
                else:
                    cache[i][j] = max(cache[i - 1][j], cache[i][j - 1])
    return cache[len(tokens1) - 1][len(tokens2) - 1]
        
def get_pt_abs_id_map(items):
    pt_abs_id_map = collections.defaultdict(set)
    
    for i in range(len(items)):
        pt, abs_id = items[i][1], items[i][5]
        pt_abs_id_map[pt].add(abs_id)
        
    return pt_abs_id_map

def train_wv_model():
    try:
        sent_tokens_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SENT_TOKENS_FILE), mode='r')
        sent_tokens = sent_tokens_file.root.data
        sent_tokens = [[w.decode() for w in tokens] for tokens in sent_tokens]
    
        model = Word2Vec(alpha=0.025, size=cnt.WV_EMBEDDING_SIZE, window=5, min_alpha=0.025, min_count=2, 
                         workers=4, negative=10, hs=0, iter=100)

        model.build_vocab(sent_tokens)
        model.train(sent_tokens, total_examples=model.corpus_count, epochs=100)
        model.save(os.path.join(cnt.DATA_FOLDER, cnt.WV_MODEL_FILE))
        
    finally:
        sent_tokens_file.close()
    
def dummy_fun(doc):
    return doc

def get_weighted_sentence_vectors(wv_model, sent_tokens, idf_scores):
    docvecs = []
    
    for tokens in sent_tokens:
        vectors, weights = [], []
        for word in tokens:
            if isinstance(word, bytes):
                word = word.decode()
                
            weights.append(idf_scores[word])
            
            if word in wv_model:
                vectors.append(wv_model[word])
            else:
                vectors.append([0] * cnt.WV_EMBEDDING_SIZE)
                
        prod = np.dot(weights, vectors) / np.sum(weights)
        docvecs.append(prod)

    return np.array(docvecs)

def get_wv_siamese(wv_model, sent_tokens):
    output = []
    for tokens in sent_tokens:
        vectors = []
        for word in tokens:
            if isinstance(word, bytes):
                word = word.decode()
                
            if word in wv_model:
                vectors.append(wv_model[word])
            else:
                vectors.append([0] * cnt.WV_EMBEDDING_SIZE)
        output.append(vectors)
    return np.array(output)
    
def generate_text_wv_embeddings():
    try:
        sent_tokens_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SENT_TOKENS_FILE), mode='r')
        sent_tokens = sent_tokens_file.root.data

        tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None, binary=True)
        tfidf_vectorizer.fit(sent_tokens)
        
        features = [x.decode() for x in tfidf_vectorizer.get_feature_names()]
        
        idf_scores = dict(zip(features, tfidf_vectorizer.idf_))
        
        wv_model = Word2Vec.load(os.path.join(cnt.DATA_FOLDER, cnt.WV_MODEL_FILE))

        embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.WV_EMBEDDINGS_FILE), mode='w')
        atom = tables.Float32Atom()
        embeds_arr = embeds_file.create_earray(embeds_file.root, 'data', atom, (0, cnt.WV_EMBEDDING_SIZE))

        n, batch_size = len(sent_tokens), cnt.PYTABLES_INSERT_BATCH_SIZE
        num_batches = int(math.ceil(float(n)/batch_size))

        for m in range(num_batches):
            start, end = m*batch_size, min((m+1)*batch_size, n)
            embeds = get_weighted_sentence_vectors(wv_model, sent_tokens[start:end,:], idf_scores)
            embeds_arr.append(embeds)

    finally:
        sent_tokens_file.close()
        embeds_file.close()
        
def get_wv_embeddings(item_indexes=None):
    try:
        embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.WV_EMBEDDINGS_FILE), mode='r')
        embeds_arr = embeds_file.root.data

        if item_indexes is not None:
            output = np.array([embeds_arr[i] for i in item_indexes])
        else:
            output = np.array([embeds_arr[i] for i in range(len(embeds_arr))])

        return output

    finally:
        embeds_file.close()