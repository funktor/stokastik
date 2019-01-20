from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from nltk.corpus import stopwords
import numpy as np
import scipy.sparse as sp

cdef extern from "FeatureTransformer.h":
    cdef struct SparseMat:
        vector[int] row, col
        vector[double] data
    
    cdef cppclass FeatureTransformer:
        vector[string] vocab
        unordered_map[string, int] vocab_inv_index
        unordered_map[string, double] idf_scores
        
        FeatureTransformer() except +
        FeatureTransformer(vector[string] stop_words, int vocab_size, int min_ngram, int max_ngram) except +
        void fit(vector[string] sentences, vector[string] labels)
        SparseMat transform(vector[string] sentences)
        
cdef class AreaRugTransformer(object):
    cdef FeatureTransformer ft
    
    def __cinit__(self, int num_features=100, int min_ngram=1, int max_ngram=3):
        self.ft = FeatureTransformer(stopwords.words('english'), num_features, min_ngram, max_ngram)
        
    def fit(self, sentences, labels):
        self.ft.fit(sentences, labels)
        
    def transform(self, sentences):
        cdef SparseMat out = self.ft.transform(sentences)
        cdef int n, m
        
        n = len(sentences)
        m = len(self.ft.vocab)
        
        return sp.csc_matrix((out.data, (out.row, out.col)), shape=(n, m))