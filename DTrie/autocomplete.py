from min_heap import MinHeap
import numpy as np, string, random, time
import sys, pickle, redis, uuid
import logging, queue, threading
from trie import TrieInterface
from trie_redis import TrieRedisInterface
from prefix_dict import SimplePrefixDict
import autocomplete_utils as utils

logging.basicConfig(filename='trie_logger.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def correctness_test(n=10000):
    strs = list(set(
        [''.join(np.random.choice(list(string.ascii_lowercase[:10]) + [' '], random.randint(0, 50))) for i in
         range(n)]))
    cnts = [random.randint(1, 10 ** 10) for i in range(len(strs))]

    trie = TrieRedisInterface(cluster_mode=False)

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


def run(q, freq_table, trie, simpl, seed_strs):
    while True:
        try:
            x, type = q.get(block=True)

            if type == 'insert':
                if freq_table.is_present(x):
                    print('Updating trie with query ', x)
                    print()
                    freq_table.add(x)
                    trie.update(x, freq_table.get(x))
                    simpl.update(x, freq_table.get(x))
                else:
                    print('Inserting into trie query ', x)
                    print()
                    freq_table.add(x)
                    trie.insert(x)
                    simpl.insert(x)

            else:
                print('Searching prefix ', x)
                out1 = trie.search(x)
                out2 = simpl.search(x)
                if sorted(out1) != sorted(out2):
                    print('!!! Mismatch !!! : ', out1, out2)
                else:
                    print('Result : ', out1)

                print()

            n = len(seed_strs)

            u = random.randint(0, 3*n-1)
            v = random.randint(0, 1)

            if u < n:
                y = seed_strs[u]
            else:
                y = ''.join(np.random.choice(list(string.ascii_lowercase[:10]), random.randint(1, 10)))

            if v == 0:
                q.put([y, 'insert'])
            else:
                k = random.randint(1, min(5, len(y)))
                pref = y[:k]
                q.put([pref, 'search'])

            time.sleep(1.0)

        except Exception as e:
            logger.error(e)


def multithread_run(num_threads=100, n=100):
    strs = list(set(
        [''.join(np.random.choice(list(string.ascii_lowercase[:10]), random.randint(1, 10))) for i in
         range(n)]))

    q = queue.Queue()
    freq_table = utils.SharedFreqTable()

    for x in strs:
        q.put([x, 'insert'])

    threads = [None] * num_threads

    trie = TrieRedisInterface(cluster_mode=False)
    simpl = SimplePrefixDict()

    for i in range(num_threads):
        print("Starting thread = ", i)
        threads[i] = threading.Thread(target=run, args=(q, freq_table, trie, simpl, strs))
        threads[i].start()

    print()

    for i in range(num_threads):
        if threads[i]:
            threads[i].join()
            print("Completed thread = ", i)


if __name__ == "__main__":
    # correctness_test(int(sys.argv[1]))
    multithread_run(10, 10)

