import requests, re, urllib
import numpy as np
import time, random
import threading, multiprocessing
from fake_useragent import UserAgent


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