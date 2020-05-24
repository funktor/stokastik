import requests, re, urllib
from bs4 import BeautifulSoup
import pandas as pd, numpy as np
import time, math, random
from multiprocessing import Process, Queue, Pool, Manager
import threading, json
import sys
import logging
from fake_useragent import UserAgent
import crawler_utils as utils
import redis
from cassandra.cluster import Cluster
import datetime

logging.basicConfig(filename='crawler.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def add_to_url_queue(rdis, bloom, out_queue, session, insert_stmt, throttle, proxies_list_sample_obj, ua, lock, max_level=5,
                     max_urls_per_page=10, use_bloom=True):

    while True:
        try:
            out = rdis.blpop('task_queue', 5.0)

            if out is None:
                break

            _, task = out

            x = json.loads(task)
            q_url, level, parent_url_hash = x['url'], int(x['level']), x['parent_url_hash']

            url_hash = hash(q_url)

            user_agent = ua.random

            headers = {"User-Agent": user_agent,
                       "Accept-Encoding": "gzip, deflate",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT": "1",
                       "Connection": "close", "Upgrade-Insecure-Requests": "1"}

            random_proxy = proxies_list_sample_obj.get_one()
            proxies = {'http': 'http://' + random_proxy}

            throttle.wait(q_url)

            r = requests.get(q_url, headers=headers, proxies=proxies)

            if r.status_code == 200:
                out_queue.put([q_url, url_hash, 'NA', parent_url_hash])

                session.execute_async(insert_stmt,
                                      [q_url, str(url_hash), 'NA', str(parent_url_hash),
                                       int(datetime.datetime.now().timestamp() * 1000)])

                if level+1 <= max_level:
                    soup = BeautifulSoup(r.content, "lxml")

                    crawled_urls = []

                    for d in soup.findAll('a', href=True):
                        if re.match('^\/wiki\/[^\/\:\.]+$', d['href']):
                            url = 'https://en.wikipedia.org' + d['href']
                            crawled_urls.append(url)

                    if len(crawled_urls) <= max_urls_per_page:
                        crawled_samples = crawled_urls
                    else:
                        crawled_weights = [1.0] * max_urls_per_page + [0.1] * (len(crawled_urls) - max_urls_per_page)
                        sample = utils.Sample(crawled_urls, crawled_weights, False)
                        crawled_samples = sample.get(max_urls_per_page)


                    if use_bloom:
                        for url in crawled_samples:
                            lock.acquire_write()
                            with rdis.pipeline() as pipe:
                                if bloom.is_present(url) is False:
                                    rdis.rpush('task_queue', json.dumps({'url': url, 'level': level + 1,
                                                                         'parent_url_hash': url_hash}))
                                    bloom.insert_key(url)
                                pipe.execute()
                            lock.release_write()

                    else:
                        for url in crawled_samples:
                            p = hash(url)
                            with rdis.pipeline() as pipe:
                                try:
                                    pipe.watch(p)

                                    if rdis.exists(p) == 0:
                                        pipe.multi()
                                        rdis.rpush('task_queue', json.dumps({'url': url, 'level': level+1,
                                                                             'parent_url_hash': url_hash}))
                                        rdis.set(p, 1)
                                        pipe.execute()

                                except redis.WatchError as e:
                                    logger.warning(e)
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    urls, url_hashes, url_text, parent_url_hash = [], [], [], []

    ua = UserAgent()

    proxies_list = utils.get_free_proxy_list()

    if len(proxies_list) <= 10:
        proxies_list_weights = [1.0] * 10
    else:
        proxies_list_weights = [1.0]*10 + [0.25]*(len(proxies_list)-10)

    proxies_list_sample_obj = utils.Sample(proxies_list, proxies_list_weights)

    m = Manager()
    out_queue = m.Queue()

    seed_url = 'https://en.wikipedia.org/wiki/Cache-oblivious_algorithm'

    r = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)
    r.rpush('task_queue', json.dumps({'url': seed_url, 'level': 0, 'parent_url_hash': ''}))
    r.set(hash(seed_url), 1)

    bloom = utils.BloomFilter(r, m=10000121, k=5)
    bloom.insert_key(seed_url)

    cluster = Cluster()
    session = cluster.connect('wiki_crawler')

    session.execute('CREATE TABLE IF NOT EXISTS crawler(url_hash text PRIMARY KEY, url text, url_text text, parent_url_hash text, inserted_time timestamp);')

    insert_stmt = session.prepare("INSERT INTO crawler(url, url_hash, url_text, parent_url_hash, inserted_time) VALUES (?, ?, ?, ?, ?)")

    max_threads, max_level, max_urls_per_page, use_bloom = 100, 3, 10, True

    throttle = utils.Throttle(1.0)

    lock = utils.ReadWriteLock()

    threads = [None] * max_threads

    for i in range(max_threads):
        print("Starting thread = ", i)
        threads[i] = threading.Thread(target=add_to_url_queue, args=(r, bloom, out_queue, session,
                                                                     insert_stmt, throttle,
                                                                     proxies_list_sample_obj, ua, lock,
                                                                     max_level, max_urls_per_page, use_bloom))
        threads[i].start()

    print()

    for i in range(max_threads):
        if threads[i]:
            threads[i].join()
            print("Completed thread = ", i)

    r.flushall()
    r.close()
    session.shutdown()

    while out_queue.empty() is not True:
        queue_top = out_queue.get()

        urls.append(queue_top[0])
        url_hashes.append(queue_top[1])
        url_text.append(queue_top[2])
        parent_url_hash.append(queue_top[3])

    df = pd.DataFrame({'URL': urls, 'URL Hash': url_hashes, 'URL Text': url_text, 'Parent URL Hash': parent_url_hash})
    df.to_csv('wiki_crawl.csv', index=False, encoding='utf-8')