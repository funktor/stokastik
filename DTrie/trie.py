from min_heap import MinHeap
import numpy as np, string, random, time
import sys, pickle, redis, uuid
import logging
from autocomplete_utils import ReadWriteLock

logging.basicConfig(filename='trie_logger.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


class Trie(object):
    def __init__(self, val='#', top_k=10):
        self.val = val
        self.top_k = top_k
        self.top_k_frequent_queries = MinHeap([], max_size=top_k)
        self.char_map = {}

    def print_trie(self):
        print(self.val, self.top_k_frequent_queries.arr)

        for _, x in self.char_map.items():
            x.print_trie()

    def get_size(self):
        s = sys.getsizeof(self.val) + sys.getsizeof(self.top_k) + sys.getsizeof(
            self.top_k_frequent_queries.arr) + sys.getsizeof(self.top_k_frequent_queries.arr_pos_map)

        for x in self.char_map:
            s += sys.getsizeof(x) + self.char_map[x].get_size()

        return s

    def update_top_k(self, curr_query, orig_query, count, trie_root):
        try:
            if len(curr_query) == 0:
                if '<END>' in self.char_map:
                    self.char_map['<END>'].top_k_frequent_queries.pop_n_push((count, orig_query))
                else:
                    trie_root.insert(orig_query, orig_query)

            else:
                q = curr_query[0]

                if q in self.char_map:
                    k = self.char_map[q].val

                    if len(k) <= len(curr_query) and curr_query[:len(k)] == k:
                        h = self.char_map[q].top_k_frequent_queries.arr_pos_map

                        if orig_query in h:
                            self.char_map[q].top_k_frequent_queries.update_key(orig_query, count)
                        else:
                            self.char_map[q].top_k_frequent_queries.pop_n_push((count, orig_query))

                        self.char_map[q].update_top_k(curr_query[len(k):], orig_query, count, trie_root)

                    else:
                        trie_root.insert(orig_query, orig_query)
                else:
                    trie_root.insert(orig_query, orig_query)

        except Exception as e:
            logger.error(e)

    def insert(self, curr_query, orig_query):
        try:
            if len(curr_query) == 0:
                self.char_map['<END>'] = Trie('#')
                self.char_map['<END>'].top_k_frequent_queries.pop_n_push((1, orig_query))

            else:
                q = curr_query[0]

                if q not in self.char_map:
                    self.char_map[q] = Trie(curr_query)
                    self.char_map[q].top_k_frequent_queries.pop_n_push((1, orig_query))
                    self.char_map[q].insert('', orig_query)

                else:
                    h1 = self.char_map[q].top_k_frequent_queries.arr[:]
                    h2 = dict(self.char_map[q].top_k_frequent_queries.arr_pos_map)

                    self.char_map[q].top_k_frequent_queries.pop_n_push((1, orig_query))

                    i = 0
                    k = self.char_map[q].val
                    while i < len(curr_query) and i < len(k) and curr_query[i] == k[i]:
                        i += 1

                    if i < len(k):
                        self.char_map[q].val = k[:i]

                        d = dict(self.char_map[q].char_map)
                        self.char_map[q].char_map = {k[i]: Trie(k[i:])}

                        self.char_map[q].char_map[k[i]].top_k_frequent_queries.arr = h1
                        self.char_map[q].char_map[k[i]].top_k_frequent_queries.arr_pos_map = h2

                        self.char_map[q].char_map[k[i]].char_map = d

                    self.char_map[q].insert(curr_query[i:], orig_query)

        except Exception as e:
            logger.error(e)

    def autocomplete_search(self, prefix):
        out = []

        try:
            if len(prefix) == 0:
                if '<END>' in self.char_map and len(self.char_map['<END>'].top_k_frequent_queries.arr) > 0:
                    out = [q for (x, q) in self.char_map['<END>'].top_k_frequent_queries.arr]

            else:
                p = prefix[0]

                if p in self.char_map:
                    k = self.char_map[p].val

                    if len(prefix) <= len(k):
                        if k[:len(prefix)] == prefix and len(self.char_map[p].top_k_frequent_queries.arr) > 0:
                            out = [q for (x, q) in self.char_map[p].top_k_frequent_queries.arr]

                    elif prefix[:len(k)] == k:
                        out = self.char_map[p].autocomplete_search(prefix[len(k):])

        except Exception as e:
            logger.error(e)

        finally:
            return out



class TrieInterface(object):
    def __init__(self):
        self.trie = Trie()
        self.lock = ReadWriteLock()

    def insert(self, query):
        self.lock.acquire_write()
        self.trie.insert(query, query)
        self.lock.release_write()

    def update(self, query, count):
        self.lock.acquire_write()
        self.trie.update_top_k(query, query, count, self.trie)
        self.lock.release_write()

    def search(self, prefix):
        self.lock.acquire_read()
        out = self.trie.autocomplete_search(prefix)
        self.lock.release_read()

        return out

    def print_trie(self):
        self.trie.print_trie()

    def get_size(self):
        return self.trie.get_size()

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.trie, f)
            f.close()

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.trie = pickle.load(f)
            f.close()