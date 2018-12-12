from siamese_network import SiameseNet
import random
from sklearn.cross_validation import train_test_split
import common_utils as utils
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
from pymongo import MongoClient 

class SiameseAPI(object):
    def __init__(self, items, groups, max_num_tokens=200, test_pct=0.2, model_path='siamese_net.h5', embedding_size=64):
        self.items = items
        self.groups = groups
        self.max_num_tokens= max_num_tokens
        self.test_pct = test_pct
        self.model_path = model_path
        self.model = None
        self.embedding_size = embedding_size
        self.word2idx = {}
        
        self.mongo_client = MongoClient()
        self.mongo_db = self.mongo_client.grouping_and_variant
        self.mongo_coll = self.mongo_db.embeddings
        
    def padd_fn(self, x):
        return x + ['PAD_TXT']*(self.max_num_tokens - len(x)) if len(x) < self.max_num_tokens else x[:self.max_num_tokens]
    
    def word_idx(self, word):
        return self.word2idx[word] if word in self.word2idx else 0
        
    def train_model(self):
        print "Creating data pairs..."
        unique_abs_id = list(set([x for x, y in self.groups.items()]))
        random.shuffle(unique_abs_id)
        
        out = []

        for abs_id_idx in range(len(unique_abs_id)):
            abs_pd_id = unique_abs_id[abs_id_idx]
            points = self.groups[abs_pd_id]

            random.shuffle(points)

            for idx in range(len(points)-1):
                pt = points[idx]
                
                positive_pt = points[idx+1]
                out.append((pt, positive_pt, 1))

                if abs_id_idx < len(unique_abs_id)-1:
                    negative_abs_id = unique_abs_id[abs_id_idx+1]
                    negative_pt = random.choice(self.groups[negative_abs_id])

                    out.append((pt, negative_pt, 0))
        
        sents1, sents2, labels = [], [], []

        for i, j, label in out:
            sent1 = self.padd_fn(utils.get_tokens(self.items[i][2]))
            sent2 = self.padd_fn(utils.get_tokens(self.items[j][2]))

            sents1.append(sent1)
            sents2.append(sent2)
            labels.append(label)
        
        print "Setting up train-test data for modeling..."
        train_indices, test_indices = train_test_split(range(len(labels)), test_size=self.test_pct)
        
        vocab = sorted(list(set([token for sent in sents1+sents2 for token in sent])))

        self.word2idx = {w: i + 1 for i, w in enumerate(vocab)}

        sent_seq1 = np.asarray([[self.word_idx(w) for w in s] for s in sents1])
        sent_seq2 = np.asarray([[self.word_idx(w) for w in s] for s in sents2])

        labels = np.asarray(labels)

        sent_seq1_tr, sent_seq1_te = sent_seq1[train_indices], sent_seq1[test_indices]
        sent_seq2_tr, sent_seq2_te = sent_seq2[train_indices], sent_seq2[test_indices]

        labels_tr, labels_te = labels[train_indices], labels[test_indices]
        
        print "Building model..."
        self.model = SiameseNet(vocab_size=len(vocab), max_words=self.max_num_tokens, embedding_size=self.embedding_size, model_file_path=self.model_path)

        self.model.build_model()
        self.model.fit(sent_seq1_tr, sent_seq2_tr, labels_tr)
        self.model.save()
        
        print "Testing model..."
        print self.model.score(sent_seq1_te, sent_seq2_te, labels_te)
    
    def get_representations(self, sentences):
        print "Getting representations..."
        sents = [self.padd_fn(utils.get_tokens(sent)) for sent in sentences]
        sent_seq = np.asarray([[self.word_idx(w) for w in s] for s in sents])
        
        return self.model.get_embeddings(sent_seq)
    
    def insert_embeddings(self):
        embeddings = self.get_representations([str(x[2]) for x in np.asarray(self.items)])
        
        print "Inserting embeddings..."
        mongo_docs = []
        for i, embed in enumerate(embeddings):
            mongo_docs.append({"item_index":i, "embedding":embed.tolist()})
        
        self.mongo_coll.insert_many(mongo_docs)
        
        print "Creating index..."
        self.mongo_coll.create_index([('item_index', pymongo.ASCENDING)], unique=True)
        
        print "Insertion complete..."
        
    def fetch_embeddings(self, item_indexes):
        print "Fetching embeddings..."
        out_embeds = dict()
        all_items = self.mongo_coll.find({"item_index": {"$in": item_indexes}})
        
        for item in all_items:
            out_embeds[item['item_index']] = item['embedding']
        
        return np.array([out_embeds[x] for x in item_indexes])
    
    def get_nearest_neighbors(self, sentence, auto_groups, head_thres=0.5, ind_thres=0.999):
        print "Getting nearest items..."
        
        heads = [x for x, y in auto_groups.items()]
        head_embeddings = self.get_representations([str(x[2]) for x in np.asarray(self.items)[heads]])
        q_embedding = self.get_representations([sentence])
        
        results = cosine_similarity(q_embedding, head_embeddings)[0]
        results = zip(range(len(results)), results)
        
        best_heads = [heads[x] for x, y in results if y >= head_thres]
        
        best_head_indexes = []
        for x in best_heads:
            best_head_indexes += auto_groups[x]
        
        best_head_embeddings = self.get_representations([str(x[2]) for x in np.asarray(self.items)[best_head_indexes]])
        
        results = cosine_similarity(q_embedding, best_head_embeddings)[0]
        results = zip(range(len(results)), results)
        
        return [self.items[best_head_indexes[x]][2] for x, y in results if y >= ind_thres]
    
    def get_nearest_neighbors_mongo(self, sentence, auto_groups, head_thres=0.5, ind_thres=0.999):
        print "Getting nearest items..."
        
        heads = [x for x, y in auto_groups.items()]
        head_embeddings = self.fetch_embeddings(heads)
        
        q_embedding = self.get_representations([sentence])
        
        results = cosine_similarity(q_embedding, head_embeddings)[0]
        results = zip(range(len(results)), results)
        
        best_heads = [heads[x] for x, y in results if y >= head_thres]
        
        best_head_indexes = []
        for x in best_heads:
            best_head_indexes += auto_groups[x]
        
        best_head_embeddings = self.fetch_embeddings(best_head_indexes)
        
        results = cosine_similarity(q_embedding, best_head_embeddings)[0]
        results = zip(range(len(results)), results)
        
        return [self.items[best_head_indexes[x]][2] for x, y in results if y >= ind_thres]