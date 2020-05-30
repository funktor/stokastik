from min_heap import MinHeap
import numpy as np, string, random, time
import sys, pickle, redis, uuid
import logging

logging.basicConfig(filename='trie_logger.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


class SimplePrefixDict(object):
    def __init__(self, top_k=10):
        self.prefixes = {}
        self.top_k = top_k

    def get_size(self):
        s = sys.getsizeof(self.top_k)
        for p in self.prefixes:
            s += sys.getsizeof(self.prefixes[p].arr) + sys.getsizeof(self.prefixes[p].arr_pos_map) + sys.getsizeof(p)
        return s

    def insert(self, query):
        p = ''
        for i in range(len(query)):
            p += query[i]

            if p not in self.prefixes:
                self.prefixes[p] = MinHeap([(1, query)], self.top_k)
            else:
                if query in self.prefixes[p].arr_pos_map:
                    self.prefixes[p].update_key(query, 1)
                else:
                    self.prefixes[p].pop_n_push((1, query))

    def update(self, query, count):
        p = ''
        for i in range(len(query)):
            p += query[i]

            if p not in self.prefixes:
                self.prefixes[p] = MinHeap([(count, query)], self.top_k)
            else:
                if query in self.prefixes[p].arr_pos_map:
                    self.prefixes[p].update_key(query, count)
                else:
                    self.prefixes[p].pop_n_push((count, query))

    def search(self, prefix):
        if prefix in self.prefixes:
            return [q for (x, q) in self.prefixes[prefix].arr]
        return []

    def save(self, file_path, s_dict):
        with open(file_path, 'wb') as f:
            pickle.dump(s_dict, f)
            f.close()

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            s_dict = pickle.load(f)
            f.close()
        return s_dict