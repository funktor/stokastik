import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import common_utils as utils
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
from pymongo import MongoClient 

class Word2VecAPI(object):
    def __init__(self, items, embedding_size=128, model_path='word2vec.pkl'):
        self.items = items
        self.model_path = model_path
        self.model = None
        self.embedding_size = embedding_size
        self.tfidf_vectorizer = None
        
        self.mongo_client = MongoClient()
        self.mongo_db = self.mongo_client.grouping_and_variant
        self.mongo_coll = self.mongo_db.embeddings

    def word2vec(self, sent_tokens):
        print("Generating WordVectors...")
        self.model = Word2Vec(alpha=0.025, size=self.embedding_size, window=5, min_alpha=0.025, min_count=5,
                                      workers=4, negative=10, hs=0, iter=200)

        self.model.build_vocab(sent_tokens)
        self.model.train(sent_tokens, total_examples=self.model.corpus_count, epochs=200)
        
    def save(self):
        print("Saving model...")
        self.model.save(self.model_path)
        
    def load(self):
        print("Loading model...")
        self.model = Word2Vec.load(self.model_path)
        
    def get_weighted_sentence_vectors(self, tfidf_matrix, vocabulary, n):
        print("Getting TF-IDF weighted document vectors...")
        docvecs = []

        for idx in range(n):
            row_vector = tfidf_matrix[idx]
            rows, cols = row_vector.nonzero()

            weights = row_vector.data

            if len(weights) > 0:
                words = [vocabulary[x] for x in cols]
                vectors = []

                for word in words:
                    if word in self.model:
                        vectors.append(self.model[word])
                    else:
                        vectors.append([0] * self.embedding_size)

                prod = np.dot(weights, vectors) / np.sum(weights)
                docvecs.append(prod)

            else:
                docvecs.append([0] * self.embedding_size)
    
        return np.array(docvecs)
        
    def train_model(self):
        print("Training word2vec model...")
        sentences = [str(x[2]) for x in self.items]
        sent_tokens = [utils.get_tokens(sent) for sent in sentences]
        
        self.word2vec(sent_tokens)
        self.save()
        
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=utils.get_tokens, ngram_range=(1,1), stop_words='english', binary=True)
        self.tfidf_vectorizer.fit(sentences)
        
    def get_representations(self, sentences):
        print("Getting representations...")
        vocabulary = [word for word, index in self.tfidf_vectorizer.vocabulary_.items()]
        matrix = self.tfidf_vectorizer.transform(sentences)
        
        return self.get_weighted_sentence_vectors(matrix, vocabulary, len(sentences))
    
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