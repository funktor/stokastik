from min_heap import MinHeap
import numpy as np, string, random, time
import sys, pickle, redis, uuid
import logging
from autocomplete_utils import ReadWriteLock
import constants as cnt

logging.basicConfig(filename=cnt.AUTOCOMPLETE_LOG_FILE, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


class SimplePrefixDict(object):
    def __init__(self, top_k=cnt.TOP_K_RESULTS):
        self.prefixes = {}
        self.top_k = top_k
        self.lock = ReadWriteLock()

    def get_size(self):
        s = sys.getsizeof(self.top_k)
        for p in self.prefixes:
            s += sys.getsizeof(self.prefixes[p].arr) + sys.getsizeof(self.prefixes[p].arr_pos_map) + sys.getsizeof(p)
        return s

    def insert(self, query):
        self.lock.acquire_write()
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
        self.lock.release_write()

    def update(self, query, count):
        self.lock.acquire_write()
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
        self.lock.release_write()

    def search(self, prefix):
        self.lock.acquire_read()
        if prefix in self.prefixes:
            out = [q for (x, q) in self.prefixes[prefix].arr]
        else:
            out = []
        self.lock.release_read()
        return out

    def save(self, file_path, s_dict):
        with open(file_path, 'wb') as f:
            pickle.dump(s_dict, f)
            f.close()

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            s_dict = pickle.load(f)
            f.close()
        return s_dict