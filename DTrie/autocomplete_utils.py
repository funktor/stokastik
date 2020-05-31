import requests, re, urllib
import numpy as np
import time, random
import threading, multiprocessing
from fake_useragent import UserAgent
import urllib.parse
import redis
import rediscluster

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


class SharedFreqTable(object):
    def __init__(self):
        self.freq_table = {}
        self.lock = ReadWriteLock()

    def add(self, x):
        self.lock.acquire_write()
        if x not in self.freq_table:
            self.freq_table[x] = 0

        self.freq_table[x] += 1
        self.lock.release_write()

    def is_present(self, x):
        self.lock.acquire_read()
        out = x in self.freq_table
        self.lock.release_read()
        return out

    def get(self, x):
        self.lock.acquire_read()
        if x in self.freq_table:
            out = self.freq_table[x]
        else:
            out = 0
        self.lock.release_read()
        return out


class BloomFilter(object):
    def __init__(self, m=17117, k=30, name='bloom', is_counting=False, cluster_mode=True):
        if cluster_mode:
            startup_nodes = [{"host": "redis-cluster.7icodg.clustercfg.usw2.cache.amazonaws.com", "port": "6379"}]
            self.rdis = rediscluster.RedisCluster(startup_nodes=startup_nodes, decode_responses=True,
                                               skip_full_coverage_check=True)
        else:
            self.rdis = redis.StrictRedis(host='127.0.0.1', port=6379, db=0, decode_responses=True)

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

    def insert_key(self, key):
        with self.rdis.pipeline() as pipe:
            positions = self.get_index_positions(key)
            for pos in positions:
                if self.is_counting:
                    self.rdis.hincrby(self.name, pos, 1)
                else:
                    self.rdis.setbit(self.name, pos, 1)
            pipe.execute()

    def insert_keys(self, keys):
        with self.rdis.pipeline() as pipe:
            for key in keys:
                positions = self.get_index_positions(key)
                for pos in positions:
                    if self.is_counting:
                        self.rdis.hincrby(self.name, pos, 1)
                    else:
                        self.rdis.setbit(self.name, pos, 1)
            pipe.execute()

    def is_present(self, key):
        with self.rdis.pipeline() as pipe:
            positions = self.get_index_positions(key)
            if self.is_counting:
                out = []
                for pos in positions:
                    x = self.rdis.hget(self.name, pos)
                    y = 1 if x is not None else 0
                    out.append(y)
            else:
                out = [self.rdis.getbit(self.name, pos) for pos in positions]
            pipe.execute()
        return all(out)

    def delete_key(self, key):
        if self.is_counting:
            if self.is_present(key):
                with self.rdis.pipeline() as pipe:
                    positions = self.get_index_positions(key)
                    for pos in positions:
                        self.rdis.hincrby(self.name, pos, -1)
                        x = self.rdis.hget(self.name, pos)
                        if x == 0:
                            self.rdis.hdel(self.name, pos)
                    pipe.execute()
        else:
            raise Exception("Feature not available with normal bloom filter")

    def delete_keys(self, keys):
        if self.is_counting:
            for key in keys:
                self.delete_key(key)
        else:
            raise Exception("Feature not available with normal bloom filter")