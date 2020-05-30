from min_heap import MinHeap
import numpy as np, string, random, time
import sys, pickle, redis, uuid
import logging
from trie import TrieInterface
from trie_redis import TrieRedisInterface
from prefix_dict import SimplePrefixDict

def correctness_test(n=10000):
    strs = list(set(
        [''.join(np.random.choice(list(string.ascii_lowercase[:10]) + [' '], random.randint(0, 50))) for i in
         range(n)]))
    cnts = [random.randint(1, 10 ** 10) for i in range(len(strs))]

    trie = TrieRedisInterface()

    start = time.time()
    for x in strs:
        trie.insert(x)
    print('Trie construction time = ', time.time() - start)

    start = time.time()
    for i in range(len(strs)):
        trie.update(strs[i], cnts[i])
    print('Trie updation time = ', time.time() - start)

    prefixes = set()
    for i in range(len(strs)):
        x = strs[i]
        p = ''
        for j in range(len(x)):
            p += x[j]
            prefixes.add(p)

    print('Number of prefixes = ', len(prefixes))

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
        trie.search(p)
    print('Trie search time = ', time.time() - start)

    start = time.time()
    for p in prefixes:
        simpl.search(p)
    print('Simple prefix dict search time = ', time.time() - start)

    # print('Size of Trie in-memory (MB) = ', trie.get_size()/1024.0**2)
    # print('Size of Simple dict in-memory (MB) = ', simpl.get_size()/1024.0**2)

    for p in prefixes:
        y = sorted(simpl.search(p))
        x = sorted(trie.search(p))

        if x != y:
            print(p)
            print(x)
            print(y)


if __name__ == "__main__":
    correctness_test(int(sys.argv[1]))

