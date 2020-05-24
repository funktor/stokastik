import requests, re, urllib
import numpy as np
import time, random
import threading, multiprocessing
from fake_useragent import UserAgent
import urllib.parse
import redis

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


class Throttle(object):
    def __init__(self, delay, is_threaded=True):
        self.lock = ReadWriteLock(is_threaded)
        self.delay = delay
        self.last_accessed_time_domain = {}
        self.event = threading.Event()

    def wait(self, url):
        self.lock.acquire_read()

        domain = urllib.parse.urlparse(url).netloc
        last_accessed = self.last_accessed_time_domain[domain] if domain in self.last_accessed_time_domain else None

        if self.delay > 0 and last_accessed is not None:
            sleep_secs = self.delay - (time.time() - last_accessed)
            if sleep_secs > 0:
                self.event.wait(sleep_secs)

        self.lock.release_read()

        self.lock.acquire_write()
        self.last_accessed_time_domain[domain] = time.time()
        self.lock.release_write()


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
    def __init__(self, is_threaded=True):
        self.shared_set = set()
        self.lock = ReadWriteLock(is_threaded)

    def get(self, item):
        self.lock.acquire_read()
        out = item in self.shared_set
        self.lock.release_read()

        return out

    def add(self, item):
        self.lock.acquire_write()
        self.shared_set.add(item)
        self.lock.release_write()


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
    def __init__(self, rdis, m=17117, k=30, name='bloom'):
        self.rdis = rdis
        self.m = m
        self.k = k
        self.name = name
        self.a1, self.b1, self.a2, self.b2 = 79, 23, 73, 61

    def get_index_positions(self, key):
        x = hash(key)
        positions = [-1]*self.k

        for i in range(self.k):
            h1 = (self.a1*x + self.b1) % self.m
            h2 = (self.a2*x + self.b2) % self.m

            positions[i] = (h1 + (i+1)*h2) % self.m

        return positions

    def insert_key(self, key):
        with self.rdis.pipeline() as pipe:
            positions = self.get_index_positions(key)
            for pos in positions:
                self.rdis.setbit(self.name, pos, 1)
            pipe.execute()

    def insert_keys(self, keys):
        with self.rdis.pipeline() as pipe:
            for key in keys:
                positions = self.get_index_positions(key)
                for pos in positions:
                    self.rdis.setbit(self.name, pos, 1)
            pipe.execute()

    def is_present(self, key):
        with self.rdis.pipeline() as pipe:
            positions = self.get_index_positions(key)
            out = [self.rdis.getbit(self.name, pos) for pos in positions]
            pipe.execute()
        return all(out)