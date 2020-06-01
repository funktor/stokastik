from min_heap import MinHeap
import numpy as np, string, random, time
import sys, pickle, redis, uuid
import logging, queue, threading
from trie import TrieInterface
from trie_redis import TrieRedisInterface
from prefix_dict import SimplePrefixDict
import autocomplete_utils as utils
import constants as cnt

logging.basicConfig(filename=cnt.AUTOCOMPLETE_LOG_FILE, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def correctness_test(n=10000):
    strs = list(set(
        [''.join(np.random.choice(list(string.ascii_lowercase[:10]) + [' '], random.randint(0, 50))) for i in
         range(n)]))
    cnts = [random.randint(1, 10 ** 10) for i in range(len(strs))]

    trie = TrieRedisInterface(cluster_mode=cnt.CLUSTER_MODE)

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


def run(q, freq_table, trie, seed_strs):
    while True:
        try:
            x, type = q.get(block=True)

            if type == 'insert':
                if freq_table.is_present(x):
                    print('Updating trie with query ', x)
                    print()
                    freq_table.add(x)
                    trie.update(x, freq_table.get(x))
                else:
                    print('Inserting into trie query ', x)
                    print()
                    freq_table.add(x)
                    trie.insert(x)

            else:
                print('Searching prefix ', x)
                out = trie.search(x)
                print('Result : ', out)

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

    trie = TrieRedisInterface(cluster_mode=cnt.CLUSTER_MODE)

    for i in range(num_threads):
        print("Starting thread = ", i)
        threads[i] = threading.Thread(target=run, args=(q, freq_table, trie, strs))
        threads[i].start()

    print()

    for i in range(num_threads):
        if threads[i]:
            threads[i].join()
            print("Completed thread = ", i)


def test_redlock(num_threads=100):
    threads = [None] * num_threads
    r = utils.get_redis_connection(cnt.CLUSTER_MODE)

    def update(rdis, my_lock):
        my_lock.lock(ttl=1000, retry_delay=200, timeout=10000, is_blocking=True)
        x = int(rdis.get('test_key'))
        rdis.set('test_key', x+1)
        my_lock.unlock()

    r.set('test_key', 0)
    lock = utils.CustomRedLock('redlock_test', cluster_mode=cnt.CLUSTER_MODE)

    for i in range(num_threads):
        print("Starting thread = ", i)
        threads[i] = threading.Thread(target=update, args=(r,lock))
        threads[i].start()

    print()

    for i in range(num_threads):
        if threads[i]:
            threads[i].join()
            print("Completed thread = ", i)

    x = int(r.get('test_key'))

    print(x)

    assert x == num_threads


if __name__ == "__main__":
    # correctness_test(int(sys.argv[1]))
    # multithread_run(100, 10)
    test_redlock(int(sys.argv[1]))