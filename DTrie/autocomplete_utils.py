import requests, re, urllib
import numpy as np
import time, random
import threading, multiprocessing
from fake_useragent import UserAgent
import urllib.parse
import redis, uuid
import rediscluster, redlock
import logging
import constants as cnt

logging.basicConfig(filename=cnt.AUTOCOMPLETE_LOG_FILE, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

class DistLock:
    def __init__(self, name='redlock', retry_count=3, retry_delay=0.2):
        self.name = name
        self.lock = False
        self.retry_delay = retry_delay
        self.retry_count = retry_count


    def acquire_lock(self, expires=1000, is_blocking=True):
        if is_blocking:
            curr_retry_delay = self.retry_delay

            while True:
                try:
                    self.red_lock = redlock.Redlock([{"host": cnt.REDIS_SERVER, "port": cnt.REDIS_PORT, "db": 0}, ],
                                                    retry_count=self.retry_count, retry_delay=curr_retry_delay)

                    self.lock = self.red_lock.lock(self.name, ttl=expires)
                    assert self.lock != False
                    return True

                except Exception as e:
                    logger.exception("error!!!")

                curr_retry_delay = min(0.2, 2*curr_retry_delay)

        else:
            self.red_lock = redlock.Redlock([{"host": cnt.REDIS_SERVER, "port": cnt.REDIS_PORT, "db": 0}, ],
                                            retry_count=self.retry_count, retry_delay=self.retry_delay)

            self.lock = self.red_lock.lock(self.name, ttl=expires)
            return True if self.lock != False else False

    def release_lock(self):
        try:
            self.red_lock.unlock(self.lock)

        except Exception as e:
            logger.exception("error!!!")


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
        self.lock = CustomRedLock('shared_freq_table_lock', cluster_mode=cnt.CLUSTER_MODE)

    def add(self, x):
        try:
            self.lock.lock(ttl=1000, retry_delay=200, timeout=10000, is_blocking=True)
            if x not in self.freq_table:
                self.freq_table[x] = 0

            self.freq_table[x] += 1
            self.lock.unlock()

        except Exception as e:
            logger.exception("error!!!")

    def is_present(self, x):
        return x in self.freq_table

    def get(self, x):
        if x in self.freq_table:
            out = self.freq_table[x]
        else:
            out = 0

        return out


class BloomFilter(object):
    def __init__(self, m=17117, k=30, name='bloom', is_counting=False, cluster_mode=True):
        self.rdis = get_redis_connection(cluster_mode)
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


def get_redis_connection(cluster_mode=False):
    if cluster_mode:
        startup_nodes = [{"host": cnt.REDIS_SERVER, "port": str(cnt.REDIS_PORT)}]
        r = rediscluster.RedisCluster(startup_nodes=startup_nodes, decode_responses=True,
                                      skip_full_coverage_check=True)
    else:
        r = redis.StrictRedis(host=cnt.REDIS_SERVER, port=cnt.REDIS_PORT, db=0, decode_responses=True)

    return r


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