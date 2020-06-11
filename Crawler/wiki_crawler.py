import requests, re, urllib
from bs4 import BeautifulSoup
import pandas as pd, numpy as np
import time, math, random
from multiprocessing import Process, Queue, Pool, Manager
import threading, json
import logging
from fake_useragent import UserAgent
import crawler_utils as utils
import datetime
from cassandra import ConsistencyLevel
import constants as cnt

logging.basicConfig(filename=cnt.LOGGER, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def add_to_url_queue(rdis, bloom, session, insert_stmt, throttle, proxies_list_sample_obj, ua, lock,
                     max_urls_per_page=10, use_bloom=True):

    while True:
        try:
            out = rdis.bzpopmax(cnt.ELASTICACHE_QUEUE_KEY, cnt.REDIS_BLOCKING_TIMEOUT)

            if out is None:
                break

            _, task, score = out

            x = json.loads(task)
            q_url, level, parent_url_hash = x['url'], int(x['level']), x['parent_url_hash']

            url_hash = utils.get_hash(q_url)

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
                soup = BeautifulSoup(r.content, "lxml")

                content = []
                for d in soup.findAll('p'):
                    if d is not None:
                        content += [utils.sanitize(d.text)]

                content = ' '.join(content)
                content = re.sub("\s\s+", " ", content)

                session.execute_async(insert_stmt,
                                      [q_url, url_hash, content, str(parent_url_hash),
                                       int(datetime.datetime.now().timestamp() * 1000)])

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
                        view = utils.get_pageviews(url, headers, proxies)
                        comp = json.dumps({'url': url, 'level': level + 1, 'parent_url_hash': url_hash})

                        lock.lock(ttl=1000, retry_delay=200, timeout=10000, is_blocking=True)

                        try:
                            if bloom.is_present(url) is False:
                                pipe = rdis.pipeline()

                                if rdis.zcard(cnt.ELASTICACHE_QUEUE_KEY) < cnt.MAXIMUM_QUEUE_SIZE:
                                    pipe.zadd(cnt.ELASTICACHE_QUEUE_KEY, {comp: view})
                                    pipe = bloom.insert_key(url, pipe)
                                    pipe.execute()

                                else:
                                    if rdis.zrange(cnt.ELASTICACHE_QUEUE_KEY, 0, 0, withscores=True)[0][1] < view:
                                        pipe.zpopmin(cnt.ELASTICACHE_QUEUE_KEY)
                                        pipe.zadd(cnt.ELASTICACHE_QUEUE_KEY, {comp: view})
                                        pipe = bloom.insert_key(url, pipe)
                                        pipe.execute()

                        except Exception as e:
                            logger.error(e)

                        lock.unlock()

                else:
                    for url in crawled_samples:
                        view = utils.get_pageviews(url, headers, proxies)
                        comp = json.dumps({'url': url, 'level': level + 1, 'parent_url_hash': url_hash})

                        p = utils.get_hash(url)
                        pipe = rdis.pipeline()
                        pipe.watch(p)

                        try:
                            if rdis.exists(p) == 0:
                                if rdis.zcard(cnt.ELASTICACHE_QUEUE_KEY) < cnt.MAXIMUM_QUEUE_SIZE:
                                    pipe.multi()
                                    pipe.zadd(cnt.ELASTICACHE_QUEUE_KEY, {comp: view})
                                    pipe.set(p, 1)
                                    pipe.execute()

                                else:
                                    if rdis.zrange(cnt.ELASTICACHE_QUEUE_KEY, 0, 0, withscores=True)[0][1] < view:
                                        pipe.multi()
                                        pipe.zpopmin(cnt.ELASTICACHE_QUEUE_KEY)
                                        pipe.zadd(cnt.ELASTICACHE_QUEUE_KEY, {comp: view})
                                        pipe.set(p, 1)
                                        pipe.execute()
                            else:
                                pipe.unwatch()

                        except Exception as e:
                            logger.error(e)

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

    max_threads, max_level, max_urls_per_page, use_bloom = cnt.NUM_THREADS, cnt.WIKI_MAX_LEVEL, cnt.WIKI_MAX_URLS_PER_PAGE, False

    r = utils.get_redis_connection(cnt.CLUSTER_MODE)
    bloom = utils.BloomFilter(r, m=cnt.BLOOM_FILTER_SIZE, k=cnt.BLOOM_FILTER_NUM_HASHES)

    if r.exists(cnt.ELASTICACHE_QUEUE_KEY) == 0:
        seed_url = cnt.WIKI_SEED_URL
        comp = json.dumps({'url': seed_url, 'level': 0, 'parent_url_hash': ''})
        r.zadd(cnt.ELASTICACHE_QUEUE_KEY, {comp:1})
        r.set(utils.get_hash(seed_url), 1)
        bloom.insert_key(seed_url)

    cluster = utils.get_cassandra_connection(cnt.CLUSTER_MODE)

    session = cluster.connect(cnt.WIKI_KEYSPACE_NAME)
    session.execute(cnt.WIKI_CREATE_TABLE_SQL)

    insert_stmt = session.prepare(cnt.WIKI_INSERT_PREP_STMT)
    insert_stmt.consistency_level = ConsistencyLevel.LOCAL_QUORUM

    throttle = utils.Throttle(cnt.THROTTLE_TIME)

    lock = utils.CustomRedLock('crawler_lock', cluster_mode=cnt.CLUSTER_MODE)

    threads = [None] * max_threads

    for i in range(max_threads):
        print("Starting thread = ", i)
        threads[i] = threading.Thread(target=add_to_url_queue, args=(r, bloom, session, insert_stmt, throttle,
                                                                     proxies_list_sample_obj, ua, lock,
                                                                     max_urls_per_page, use_bloom))
        threads[i].start()

    print()

    for i in range(max_threads):
        if threads[i]:
            threads[i].join()
            print("Completed thread = ", i)