from min_heap import MinHeap
import numpy as np, string, random, time
import sys, pickle, redis, uuid
import logging
from autocomplete_utils import ReadWriteLock

logging.basicConfig(filename='trie_logger.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


class TrieRedis(object):
    def __init__(self, top_k=10):
        self.top_k = top_k

    def print_trie(self, rdis, root_node_id):
        print(rdis.get(root_node_id + ':val'), rdis.zrange(root_node_id + ':top_k', 0, -1, withscores=True))

        for _, x in rdis.hgetall(root_node_id + ':children').items():
            self.print_trie(rdis, x)


    def update_top_k(self, curr_query, orig_query, count, rdis, root_node_id):
        try:
            if len(curr_query) == 0:
                node_id = rdis.hget(root_node_id + ':children', '<END>')

                if node_id is not None:
                    rdis.zadd(node_id + ':top_k', {orig_query: count})

            else:
                q = curr_query[0]
                node_id = rdis.hget(root_node_id + ':children', q)

                if node_id is not None:
                    k = rdis.get(node_id + ':val')

                    if len(k) <= len(curr_query) and curr_query[:len(k)] == k:
                        h = rdis.zscore(node_id + ':top_k', orig_query)

                        if h is not None:
                            rdis.zadd(node_id + ':top_k', {orig_query: count})
                        else:
                            if rdis.zcard(node_id + ':top_k') < self.top_k:
                                rdis.zadd(node_id + ':top_k', {orig_query: count})
                            else:
                                if rdis.zrange(node_id + ':top_k', 0, 0, withscores=True)[0][1] < count:
                                    pipe = rdis.pipeline()
                                    pipe.zpopmin(node_id + ':top_k')
                                    pipe.zadd(node_id + ':top_k', {orig_query: count})
                                    pipe.execute()

                        self.update_top_k(curr_query[len(k):], orig_query, count, rdis, node_id)

        except Exception as e:
            logger.error(e)


    def insert(self, curr_query, orig_query, rdis, root_node_id):
        try:
            if len(curr_query) == 0:
                node_id = str(uuid.uuid1())

                pipe = rdis.pipeline(False)
                pipe.set(node_id + ':val', '#')
                pipe.hset(root_node_id + ':children', '<END>', node_id)
                pipe.execute()

                if rdis.zcard(node_id + ':top_k') < self.top_k:
                    rdis.zadd(node_id + ':top_k', {orig_query: 1})

            else:
                q = curr_query[0]
                node_id = rdis.hget(root_node_id + ':children', q)

                if node_id is None:
                    node_id = str(uuid.uuid1())

                    pipe = rdis.pipeline(False)
                    pipe.set(node_id + ':val', curr_query)
                    pipe.hset(root_node_id + ':children', q, node_id)
                    pipe.execute()

                    if rdis.zcard(node_id + ':top_k') < self.top_k:
                        rdis.zadd(node_id + ':top_k', {orig_query: 1})

                    self.insert('', orig_query, rdis, node_id)

                else:
                    k = rdis.get(node_id + ':val')

                    i = 0
                    while i < len(curr_query) and i < len(k) and curr_query[i] == k[i]:
                        i += 1

                    if i < len(k):
                        new_child_id = str(uuid.uuid1())
                        d = rdis.hgetall(node_id + ':children')

                        pipe = rdis.pipeline(False)
                        pipe.set(node_id + ':val', k[:i])

                        pipe.delete(node_id + ':children')
                        pipe.hset(node_id + ':children', k[i], new_child_id)

                        pipe.set(new_child_id + ':val', k[i:])

                        pipe.zunionstore(new_child_id + ':top_k', [node_id + ':top_k'])

                        pipe.hmset(new_child_id + ':children', d)
                        pipe.execute()

                    if rdis.zcard(node_id + ':top_k') < self.top_k:
                        rdis.zadd(node_id + ':top_k', {orig_query: 1})

                    self.insert(curr_query[i:], orig_query, rdis, node_id)

        except Exception as e:
            logger.error(e)


    def autocomplete_search(self, prefix, rdis, root_node_id):
        out = []
        try:
            if len(prefix) == 0:
                node_id = rdis.hget(root_node_id + ':children', '<END>')
                if node_id is not None:
                    out = rdis.zrange(node_id + ':top_k', 0, -1)

            else:
                p = prefix[0]
                node_id = rdis.hget(root_node_id + ':children', p)

                if node_id is not None:
                    k = rdis.get(node_id + ':val')

                    if len(prefix) <= len(k):
                        if k[:len(prefix)] == prefix:
                            out = rdis.zrange(node_id + ':top_k', 0, -1)

                    elif prefix[:len(k)] == k:
                        out = self.autocomplete_search(prefix[len(k):], rdis, node_id)

        except Exception as e:
            logger.error(e)

        finally:
            return out


class TrieRedisInterface(object):
    def __init__(self):
        self.trie = TrieRedis()
        self.lock = ReadWriteLock()
        self.r = redis.StrictRedis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
        self.root_node_id = str(uuid.uuid1())
        self.r.set(self.root_node_id + ':val', '#')

    def insert(self, query):
        self.lock.acquire_write()
        self.trie.insert(query, query, self.r, self.root_node_id)
        self.lock.release_write()

    def update(self, query, count):
        self.lock.acquire_write()
        self.trie.update_top_k(query, query, count, self.r, self.root_node_id)
        self.lock.release_write()

    def search(self, prefix):
        self.lock.acquire_read()
        out = self.trie.autocomplete_search(prefix, self.r, self.root_node_id)
        self.lock.release_read()

        return out

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump([self.trie, self.root_node_id], f)
            f.close()

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.trie, self.root_node_id = pickle.load(f)
            f.close()