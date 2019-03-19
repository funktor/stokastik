import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np, tables, math
from gensim.models import Word2Vec
import data_generator as dg
from sklearn.neighbors import KDTree

def dummy_fun(doc):
    return doc

class Word2VecAPI(object):
    def __init__(self, embedding_size=256, model_path='word2vec.pkl'):
        self.model_path = model_path
        self.model = None
        self.embedding_size = embedding_size
        self.tfidf_vectorizer = None
        
    def get_model(self):
        self.model = Word2Vec(alpha=0.025, size=self.embedding_size, window=5, min_alpha=0.025, min_count=2, 
                              workers=4, negative=10, hs=0, iter=100)
        
    def save(self):
        self.model.save(self.model_path)
        
    def load(self):
        self.model = Word2Vec.load(self.model_path)
        
    def get_weighted_sentence_vectors(self, tfidf_matrix, vocabulary, n):
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
        try:
            tokens_file = tables.open_file('data/sent_tokens.h5', mode='r')
            sent_tokens = tokens_file.root.data
            
            sent_tokens = [[w.decode('utf-8') for w in tokens] for tokens in sent_tokens]
            
            self.get_model()
            self.model.build_vocab(sent_tokens)
            self.model.train(sent_tokens, total_examples=self.model.corpus_count, epochs=100)
            self.save()
            
            self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None, binary=True)
            self.tfidf_vectorizer.fit(sent_tokens)

        finally:
            tokens_file.close()
        
        
    def insert_embeddings_pytables(self, batch_size=25000):
        try:
            self.load()
            
            embeds_file = tables.open_file('data/w2v_embeddings.h5', mode='w')
            atom = tables.Float32Atom()
            embeds_arr = embeds_file.create_earray(embeds_file.root, 'data', atom, (0, self.embedding_size))
            
            tokens_file = tables.open_file('data/sent_tokens.h5', mode='r')
            sent_tokens = tokens_file.root.data
            
            sent_tokens = [[w.decode('utf-8') for w in tokens] for tokens in sent_tokens]
            
            n = len(sent_tokens)
            num_batches = int(math.ceil(float(n)/batch_size))
            
            vocabulary = [word for word, index in self.tfidf_vectorizer.vocabulary_.items()]
            
            for m in range(num_batches):
                start, end = m*batch_size, min((m+1)*batch_size, n)
                matrix = self.tfidf_vectorizer.transform(sent_tokens[start:end,:])
                embeds = self.get_weighted_sentence_vectors(matrix, vocabulary, end-start)
                embeds_arr.append(embeds)
                
        finally:
            tokens_file.close()
            embeds_file.close()
    
    def construct_kd_trees(self, groups):
        try:
            embeds_file = tables.open_file('data/w2v_embeddings.h5', mode='r')
            embeddings = embeds_file.root.data
            kdtree = KDTree(embeddings, leaf_size=50)
            
            dg.save_data_pkl(kd_tree_grps, 'kd_tree_grps_w2v.pkl')
                
        finally:
            embeds_file.close()