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