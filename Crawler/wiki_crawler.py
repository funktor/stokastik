import requests, re, urllib
from bs4 import BeautifulSoup
import pandas as pd, numpy as np
import time, math, random
from multiprocessing import Process, Queue, Pool, Manager
import threading
import sys
import logging
from fake_useragent import UserAgent
import crawler_utils as utils

logging.basicConfig(filename='crawler.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def add_to_url_queue(q_urls, queue, out_queue, throttle, proxies_list_sample_obj, ua, shared_url_set, max_level=5,
                     max_urls_per_page=10):

    for q_url, level, parent_url_hash in q_urls:
        try:
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

                    for url in crawled_samples:
                        p = hash(url)

                        if shared_url_set.get(p) is False:
                            queue.put([url, level + 1, url_hash])
                            shared_url_set.add(p)

        except Exception as e:
            logger.error(e)
            time.sleep(5.0)


def bfs(queue, out_queue, proxies_list_sample_obj, ua, shared_url_set, max_level=5, max_threads=50,
        max_urls_per_page=10):

    curr_level = 0

    while curr_level <= max_level:
        try:
            curr_queue_data = []

            while queue.empty() is not True:
                curr_queue_data.append(queue.get())

            print("Current level = ", curr_level)
            print(len(curr_queue_data))

            if len(curr_queue_data) == 0:
                break

            throttle = utils.Throttle(1.0)

            n_threads = min(max_threads, len(curr_queue_data))

            batch_size = int(math.ceil(len(curr_queue_data) / float(n_threads)))
            threads = [None] * n_threads

            for i in range(n_threads):
                print("Starting thread = ", i)
                start, end = i * batch_size, min((i + 1) * batch_size, len(curr_queue_data))
                urls = [curr_queue_data[j] for j in range(start, end)]

                threads[i] = threading.Thread(target=add_to_url_queue, args=(urls, queue, out_queue, throttle,
                                                                             proxies_list_sample_obj, ua,
                                                                             shared_url_set, max_level,
                                                                             max_urls_per_page))
                threads[i].start()

            print()

            for i in range(n_threads):
                if threads[i]:
                    threads[i].join()
                    print("Completed thread = ", i)

            curr_level += 1

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

    m1, m2 = Manager(), Manager()

    urls_queue, out_queue = m1.Queue(), m2.Queue()

    seed_url = 'https://en.wikipedia.org/wiki/Cache-oblivious_algorithm'

    urls_queue.put([seed_url, hash(seed_url), None])

    shared_url_set = utils.SharedSet()

    bfs(urls_queue, out_queue, proxies_list_sample_obj, ua, shared_url_set, max_level=2, max_threads=50,
        max_urls_per_page=20)

    while out_queue.empty() is not True:
        queue_top = out_queue.get()

        urls.append(queue_top[0])
        url_hashes.append(queue_top[1])
        url_text.append(queue_top[2])
        parent_url_hash.append(queue_top[3])

    df = pd.DataFrame({'URL': urls, 'URL Hash': url_hashes, 'URL Text': url_text, 'Parent URL Hash': parent_url_hash})
    df.to_csv('wiki_crawl.csv', index=False, encoding='utf-8')