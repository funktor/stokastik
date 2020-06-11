import requests, re, urllib
import numpy as np
import time, random
import threading, multiprocessing
from fake_useragent import UserAgent
import urllib.parse
import redis, uuid
import constants as cnt
import logging, rediscluster
from datetime import datetime, timedelta
from dateutil.relativedelta import *
from cassandra.cluster import Cluster
from ssl import SSLContext, PROTOCOL_TLSv1, CERT_REQUIRED
from cassandra.auth import PlainTextAuthProvider
import hashlib

logging.basicConfig(filename=cnt.LOGGER, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

class ReadWriteLock:
    def __init__(self, is_threaded=True):
        if is_threaded:
            self._read_ready = threading.Condition(threading.Lock())
        else:
            self._read_ready = multiprocessing.Condition(multiprocessing.Lock())

        self._readers = 0

    def acquire_read(self):
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()


def get_redis_connection(cluster_mode=False):
    if cluster_mode:
        startup_nodes = [{"host": cnt.ELASTICACHE_URL, "port": str(cnt.ELASTICACHE_PORT)}]
        r = rediscluster.RedisCluster(startup_nodes=startup_nodes, decode_responses=True,
                                      skip_full_coverage_check=True)
    else:
        r = redis.StrictRedis(host='127.0.0.1', port=cnt.ELASTICACHE_PORT, db=0, decode_responses=True)

    return r

def get_cassandra_connection(cluster_mode=False):
    if cluster_mode:
        ssl_context = SSLContext(PROTOCOL_TLSv1)
        ssl_context.load_verify_locations(cnt.AWS_KEYSPACES_PEM)
        ssl_context.verify_mode = CERT_REQUIRED
        auth_provider = PlainTextAuthProvider(username=cnt.AWS_KEYSPACES_USER,
                                              password=cnt.AWS_KEYSPACES_PASSWD)

        cluster = Cluster([cnt.CASSANDRA_URL], ssl_context=ssl_context, auth_provider=auth_provider,
                          port=cnt.CASSANDRA_PORT)
    else:
        cluster = Cluster(['127.0.0.1'], port=9042)

    return cluster


class CustomRedLock:
    def __init__(self, resource_name='redlock', cluster_mode=True):
        self.resource_name = resource_name
        self.rdis = get_redis_connection(cluster_mode)
        self.lock_random_value = str(uuid.uuid1())

    def lock(self, ttl=1000, retry_count=3, retry_delay=200, timeout=5000, is_blocking=False):
        if is_blocking:
            start = time.time()

            while True:
                try:
                    h = self.rdis.set(self.resource_name, self.lock_random_value, nx=True, px=ttl)
                    assert h is not None
                    return True

                except Exception as e:
                    logger.exception("error!!!")

                time.sleep(retry_delay/1000.0)

                if 1000*(time.time()-start) > timeout:
                    break

        else:
            start = time.time()

            for i in range(retry_count):
                try:
                    h = self.rdis.set(self.resource_name, self.lock_random_value, nx=True, px=ttl)
                    assert h is not None
                    return True

                except Exception as e:
                    logger.exception("error!!!")

                time.sleep(retry_delay / 1000.0)

                if 1000*(time.time()-start) > timeout:
                    break

        return False

    def unlock(self):
        try:
            h = self.rdis.get(self.resource_name)

            if h is not None and h == self.lock_random_value:
                self.rdis.delete(self.resource_name)
                return True

        except Exception as e:
            logger.exception("error!!!")

        return False


    def is_locked(self):
        try:
            return self.rdis.exists(self.resource_name)

        except Exception as e:
            logger.exception("error!!!")

        return False


class Throttle(object):
    def __init__(self, delay):
        self.lock = CustomRedLock('timeout_lock', cluster_mode=cnt.CLUSTER_MODE)
        self.delay = delay
        self.last_accessed_time_domain = {}

    def wait(self, url):
        try:
            self.lock.lock(ttl=1000, retry_delay=200, timeout=10000, is_blocking=True)

            domain = urllib.parse.urlparse(url).netloc
            last_accessed = self.last_accessed_time_domain[domain] if domain in self.last_accessed_time_domain else None

            if self.delay > 0 and last_accessed is not None:
                sleep_secs = self.delay - (time.time() - last_accessed)
                if sleep_secs > 0:
                    time.sleep(sleep_secs)

            self.last_accessed_time_domain[domain] = time.time()
            self.lock.unlock()

        except Exception as e:
            logger.exception("error!!!")


class Sample(object):
    def __init__(self, values, weights, with_replacement=True):
        self.values = values
        self.weights = weights

        s = sum(self.weights)
        self.weights = [x / float(s) if s != 0 else 0.0 for x in self.weights]

        self.cum_sum = [0] * len(self.values)
        self.compute_cum_sum()
        self.with_replacement = with_replacement

    def compute_cum_sum(self):
        cum_sum = [0] * len(self.values)
        s = 0
        for i in range(len(self.values)):
            s += self.weights[i]
            cum_sum[i] = s

        self.cum_sum = cum_sum

    def get_one(self):
        u = random.uniform(0, 1)

        left, right = 0, len(self.values) - 1
        last_true = 0
        while left <= right:
            mid = int((left + right) / 2)
            if self.cum_sum[mid] >= u:
                last_true = mid
                right = mid - 1
            else:
                left = mid + 1

        return self.values[last_true]

    def get(self, cnt=1):
        if self.with_replacement:
            return [self.get_one() for _ in range(cnt)]
        else:
            if cnt >= len(self.values):
                return self.values

            return np.random.choice(self.values, cnt, p=self.weights, replace=False)


class SharedSet(object):
    def __init__(self):
        self.shared_set = set()
        self.lock = CustomRedLock('shared_set_lock', cluster_mode=cnt.CLUSTER_MODE)

    def get(self, item):
        try:
            self.lock.lock(ttl=1000, retry_delay=200, timeout=10000, is_blocking=True)
            out = item in self.shared_set
            self.lock.unlock()
            return out

        except Exception as e:
            logger.exception("error!!!")

    def add(self, item):
        try:
            self.lock.lock(ttl=1000, retry_delay=200, timeout=10000, is_blocking=True)
            self.shared_set.add(item)
            self.lock.unlock()

        except Exception as e:
            logger.exception("error!!!")


def get_free_proxy_list():
    ua = UserAgent()
    user_agent = ua.random

    headers = {"User-Agent": user_agent,
               "Accept-Encoding": "gzip, deflate",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT": "1",
               "Connection": "close", "Upgrade-Insecure-Requests": "1"}

    r = requests.get('https://free-proxy-list.net', headers=headers)

    return re.findall('[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\:[0-9]+', r.text)


def clean_tokens(tokens, to_replace='[^a-zA-Z0-9\'\.\" ]+'):
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    return tokens


def tokenize(mystr):
    return mystr.lower().split(" ")


def sanitize(sentence, to_replace='[^a-zA-Z0-9-\'\.\" ]+'):
    sentence = re.sub('<[^<]+?>', ' ', sentence)
    sentence = re.sub(to_replace, ' ', sentence).strip()
    sentence = re.sub('(?<![0-9])\.(?![0-9])|(?<=[0-9])\.(?![0-9])|(?<![0-9])\.(?=[0-9])', ' ', sentence).strip()
    sentence = re.sub('\s+', ' ', sentence)

    tokens = clean_tokens(tokenize(sentence), to_replace)
    tokens = [re.sub('\s+', ' ', token) for token in tokens]

    return ' '.join([x.strip() for x in tokens])


def sieve(a, b, n):
    arr = list(range(2, b + 1))
    out = []

    while True:
        new_arr, h = [], arr[0]
        out.append(h)

        for x in arr[1:]:
            if x % h != 0:
                new_arr.append(x)

        if len(new_arr) == 0:
            break

        arr = new_arr[:]

    out = [x for x in out if x >= a]

    if len(out) <= n:
        return out

    return random.sample(out, n)


class BloomFilter(object):
    def __init__(self, rdis, m=17117, k=30, name='bloom', is_counting=False):
        self.rdis = rdis
        self.m = m
        self.k = k
        self.name = name
        self.a1, self.b1, self.a2, self.b2 = 79, 23, 73, 61
        self.is_counting = is_counting

    def get_index_positions(self, key):
        x = hash(key)
        positions = [-1]*self.k

        for i in range(self.k):
            h1 = (self.a1*x + self.b1) % self.m
            h2 = (self.a2*x + self.b2) % self.m

            positions[i] = (h1 + (i+1)*h2) % self.m

        return positions

    def insert_key(self, key, pipe=None):
        positions = self.get_index_positions(key)

        if pipe is not None:
            for pos in positions:
                if self.is_counting:
                    pipe.hincrby(self.name, pos, 1)
                else:
                    pipe.setbit(self.name, pos, 1)
            return pipe

        else:
            pipe = self.rdis.pipeline()

            for pos in positions:
                if self.is_counting:
                    pipe.hincrby(self.name, pos, 1)
                else:
                    pipe.setbit(self.name, pos, 1)

            pipe.execute()
            return 1

    def insert_keys(self, keys, pipe=None):
        if pipe is not None:
            for key in keys:
                pipe = self.insert_key(key, pipe)
            return pipe

        else:
            pipe = self.rdis.pipeline()
            for key in keys:
                positions = self.get_index_positions(key)
                for pos in positions:
                    if self.is_counting:
                        pipe.hincrby(self.name, pos, 1)
                    else:
                        pipe.setbit(self.name, pos, 1)
            pipe.execute()
            return 1

    def is_present(self, key):
        positions = self.get_index_positions(key)

        if self.is_counting:
            pipe = self.rdis.pipeline()
            for pos in positions:
                pipe.hget(self.name, pos)
            out = pipe.execute()

            out = [1 if x is not None else 0 for x in out]

        else:
            pipe = self.rdis.pipeline()
            for pos in positions:
                pipe.getbit(self.name, pos)
            out = pipe.execute()

        return all(out)

    def delete_key(self, key):
        if self.is_counting:
            if self.is_present(key):
                positions = self.get_index_positions(key)

                pipe = self.rdis.pipeline()
                for pos in positions:
                    pipe.hincrby(self.name, pos, -1)
                pipe.execute()

                pipe = self.rdis.pipeline()
                for pos in positions:
                    pipe.hget(self.name, pos)
                x = pipe.execute()

                pipe = self.rdis.pipeline()
                for i in range(len(positions)):
                    if x[i] == 0:
                        pipe.hdel(self.name, positions[i])
                pipe.execute()

        else:
            raise Exception("Feature not available with normal bloom filter")

    def delete_keys(self, keys):
        if self.is_counting:
            for key in keys:
                self.delete_key(key)
        else:
            raise Exception("Feature not available with normal bloom filter")


def get_pageviews(url, headers=None, proxies=None):
    try:
        page_title = re.sub('https://.*?\/wiki/(.*)', '\\1', url)

        curr_date = datetime.today()
        old_date = curr_date + relativedelta(months=-6)

        stat_url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/'
        stat_url += page_title
        stat_url += '/monthly/' + old_date.strftime('%Y%m%d') + '/' + curr_date.strftime('%Y%m%d')

        r_stat = requests.get(stat_url, headers=headers, proxies=proxies)
        resp = r_stat.json()

        views = 0

        if 'items' in resp:
            for x in resp['items']:
                views += x['views']

        return views

    except Exception as e:
        logger.exception("error!!!")

def get_hash(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()