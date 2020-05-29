from min_heap import MinHeap
import numpy as np, string, random, time
import sys, pickle
from autocomplete_utils import ReadWriteLock

class Trie(object):
    def __init__(self, val='#', top_k=3):
        self.val = val
        self.top_k = top_k
        self.top_k_frequent_queries = MinHeap([], max_size=top_k)
        self.char_map = {}
        self.lock = ReadWriteLock()

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

    def insert(self, curr_query, orig_query):
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

    def autocomplete_search(self, prefix):
        if len(prefix) == 0:
            if '<END>' in self.char_map and len(self.char_map['<END>'].top_k_frequent_queries.arr) > 0:
                return [q for (x, q) in self.char_map['<END>'].top_k_frequent_queries.arr]
            return []

        p = prefix[0]

        if p in self.char_map:
            k = self.char_map[p].val

            if len(prefix) <= len(k):
                if k[:len(prefix)] == prefix and len(self.char_map[p].top_k_frequent_queries.arr) > 0:
                    return [q for (x, q) in self.char_map[p].top_k_frequent_queries.arr]
                return []

            if prefix[:len(k)] != k:
                return []

            return self.char_map[p].autocomplete_search(prefix[len(k):])

    def save(self, file_path, trie):
        with open(file_path, 'wb') as f:
            pickle.dump(trie, f)
            f.close()

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            trie = pickle.load(f)
            f.close()
        return trie


class SimplePrefixDict(object):
    def __init__(self, top_k=3):
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

    def autocomplete_search(self, prefix):
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


def correctness_test(n=10000):
    strs = list(set(
        [''.join(np.random.choice(list(string.ascii_lowercase[:10]) + [' '], random.randint(0, 50))) for i in
         range(n)]))
    cnts = [random.randint(1, 10 ** 10) for i in range(len(strs))]

    trie = Trie()

    start = time.time()
    for x in strs:
        trie.insert(x, x)
    print('Trie construction time = ', time.time() - start)

    start = time.time()
    for i in range(len(strs)):
        trie.update_top_k(strs[i], strs[i], cnts[i], trie)
    print('Trie updation time = ', time.time() - start)

    prefixes = set()
    for i in range(len(strs)):
        x = strs[i]
        p = ''
        for j in range(len(x)):
            p += x[j]
            prefixes.add(p)

    simpl = SimplePrefixDict()

    start = time.time()
    for x in strs:
        simpl.insert(x)
    print('Simple prefix dict construction time = ', time.time() - start)

    start = time.time()
    for i in range(len(strs)):
        simpl.update(strs[i], cnts[i])
    print('Simple prefix dict updation time = ', time.time() - start)

    start = time.time()
    for p in prefixes:
        trie.autocomplete_search(p)
    print('Trie search time = ', time.time() - start)

    start = time.time()
    for p in prefixes:
        simpl.autocomplete_search(p)
    print('Simple prefix dict search time = ', time.time() - start)


    print('Size of Trie in-memory (MB) = ', trie.get_size()/1024.0**2)
    print('Size of Simple dict in-memory (MB) = ', simpl.get_size()/1024.0**2)

    for p in prefixes:
        y = sorted(simpl.autocomplete_search(p))
        x = sorted(trie.autocomplete_search(p))

        if x != y:
            print(p)
            print(x)
            print(y)


if __name__ == "__main__":
    correctness_test(int(sys.argv[1]))

