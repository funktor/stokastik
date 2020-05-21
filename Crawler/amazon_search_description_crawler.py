import requests, re, urllib
from bs4 import BeautifulSoup
import pandas as pd, numpy as np
import time, math, random
from multiprocessing import Process, Queue, Pool, Manager
import threading
import logging, json
from fake_useragent import UserAgent
import crawler_utils as utils
import urllib.parse


logging.basicConfig(filename='crawler.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def insert_metadata(soup, query, q_url, url_hash, level, out_queue):
    next_level_urls = []

    if level == 0:
        for d in soup.findAll('div', attrs={'class': 'sg-col-4-of-12 sg-col-8-of-16 sg-col-16-of-24 sg-col-12-of-20 sg-col-24-of-32 sg-col sg-col-28-of-36 sg-col-20-of-28'}):
            try:
                p_title = d.find('span', attrs={'class': 'a-size-medium a-color-base a-text-normal'})
                p_price = d.find('span', attrs={'class': 'a-offscreen'})
                p_rating = d.find('span', attrs={'class': 'a-icon-alt'})
                p_product_url = d.find('a', attrs={'class': 'a-link-normal a-text-normal'}, href=True)

                all = [query, q_url, url_hash]
                metadata = {}

                if p_title is not None:
                    metadata['title'] = utils.sanitize(p_title.text)
                else:
                    metadata['title'] = ''

                if p_price is not None:
                    metadata['price'] = p_price.text
                else:
                    metadata['price'] = ''

                if p_rating is not None:
                    metadata['rating'] = p_rating.text
                else:
                    metadata['rating'] = ''

                if p_product_url is not None:
                    scheme = urllib.parse.urlparse(q_url).scheme
                    domain = urllib.parse.urlparse(q_url).netloc
                    url = scheme + '://' + domain + p_product_url['href']

                    metadata['product_url'] = url
                    metadata['product_url_hash'] = hash(url)
                    next_level_urls.append(url)
                else:
                    metadata['product_url'] = ''
                    metadata['product_url_hash'] = ''

                all += [metadata, time.time()]

                out_queue.put(all)

            except Exception as e:
                logger.error(e)

        for d in soup.findAll('div', attrs={'class': 'sg-col-4-of-24 sg-col-4-of-12 sg-col-4-of-36 s-result-item s-asin sg-col-4-of-28 sg-col-4-of-16 sg-col sg-col-4-of-20 sg-col-4-of-32'}):
            try:
                p_title = d.find('span', attrs={'class': 'a-size-base-plus a-color-base a-text-normal'})
                p_price = d.find('span', attrs={'class': 'a-offscreen'})
                p_rating = d.find('span', attrs={'class': 'a-icon-alt'})
                p_product_url = d.find('a', attrs={'class': 'a-link-normal a-text-normal'}, href=True)

                all = [query, q_url, url_hash]
                metadata = {}

                if p_title is not None:
                    metadata['title'] = utils.sanitize(p_title.text)
                else:
                    metadata['title'] = ''

                if p_price is not None:
                    metadata['price'] = p_price.text
                else:
                    metadata['price'] = ''

                if p_rating is not None:
                    metadata['rating'] = p_rating.text
                else:
                    metadata['rating'] = ''

                if p_product_url is not None:
                    scheme = urllib.parse.urlparse(q_url).scheme
                    domain = urllib.parse.urlparse(q_url).netloc
                    url = scheme + '://' + domain + p_product_url['href']

                    metadata['product_url'] = url
                    metadata['product_url_hash'] = hash(url)
                    next_level_urls.append(url)
                else:
                    metadata['product_url'] = ''
                    metadata['product_url_hash'] = ''

                all += [metadata, time.time()]

                out_queue.put(all)

            except Exception as e:
                logger.error(e)
    else:
        metadata = {}

        title = soup.find('span', attrs={'id': 'productTitle'})
        if title is not None:
            metadata['title'] = utils.sanitize(title.text)
        else:
            metadata['title'] = ''

        rating = soup.find('span', attrs={'class': 'reviewCountTextLinkedHistogram noUnderline'}, title=True)
        if rating is not None:
            metadata['rating'] = rating['title']
        else:
            metadata['rating'] = ''

        price = soup.find('span', attrs={'class': 'a-size-medium a-color-price priceBlockBuyingPriceString'})
        if price is not None:
            metadata['price'] = price.text
        else:
            metadata['price'] = ''

        metadata['product_url'] = q_url
        metadata['product_url_hash'] = url_hash

        ul = soup.find('ul', attrs={'class': 'a-unordered-list a-vertical a-spacing-mini'})
        description = []

        if ul is not None:
            for d in ul.findAll('span', attrs={'class': 'a-list-item'}):
                if d is not None:
                    description += [d.text]

        metadata['description'] = utils.sanitize(' '.join(description))

        all = [query, q_url, url_hash, metadata, time.time()]
        out_queue.put(all)

    return next_level_urls


def add_to_url_queue(q_urls, queue, out_queue, throttle, proxies_list_sample_obj, ua, max_level=5):

    for query, q_url, level in q_urls:
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
                soup = BeautifulSoup(r.content, "lxml")

                next_level_urls = insert_metadata(soup, query, q_url, url_hash, level, out_queue)

                if len(next_level_urls) > 0 and level+1 <= max_level:
                    for url in next_level_urls:
                        queue.put([query, url, level + 1])

        except Exception as e:
            logger.error(e)


def bfs(queue, out_queue, proxies_list_sample_obj, ua, max_level=5, max_threads=50):
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
                                                                             proxies_list_sample_obj, ua, max_level))
                threads[i].start()

            print()

            for i in range(n_threads):
                if threads[i]:
                    threads[i].join()
                    print("Completed thread = ", i)

            curr_level += 1

        except Exception as e:
            logger.error(e)


def get_urls(queries, page_nums_sample_obj, domains_sample_obj, max_page_num=5):
    page_nums = page_nums_sample_obj.get(max_page_num)

    urls = []
    for query in queries:
        for page_num in page_nums:
            qs = urllib.parse.urlencode({'k' : query, 'page' : str(page_num)})
            rand_domain = domains_sample_obj.get_one()
            urls.append((query, 'https://' + rand_domain + '/s?' + qs, 0))

    random.shuffle(urls)
    return urls


if __name__ == "__main__":
    max_threads = 50
    max_page_num = 20

    search_queries, urls, url_hashes, metadata, timestamp = [], [], [], [], []

    domains = ['www.amazon.com',
               'www.amazon.in',
               'www.amazon.co.uk',
               'www.amazon.fr',
               'www.amazon.au',
               'www.amazon.sg']

    domain_weights = [1.0, 0.50, 0.25, 0.20, 0.20, 0.20]

    domains_sample_obj = utils.Sample(domains, domain_weights)

    queries = ['Laptops',
               'Mobile Phones',
               'Iphone',
               'Microwave',
               'Headphones',
               'Televisions',
               'Mattresses',
               'T-shirts',
               'USB Cable',
               'Programming Books',
               'Bras',
               'Area Rug',
               'Curtains',
               'Monitors']

    ua = UserAgent()

    proxies_list = utils.get_free_proxy_list()

    if len(proxies_list) <= 10:
        proxies_list_weights = [1.0] * 10
    else:
        proxies_list_weights = [1.0] * 10 + [0.25] * (len(proxies_list) - 10)

    proxies_list_sample_obj = utils.Sample(proxies_list, proxies_list_weights)

    page_nums_sample_obj = utils.Sample(list(range(1, 2 * max_page_num + 1)),
                                        [1.0] * max_page_num + [0.2] * max_page_num,
                                        False)

    q_urls = get_urls(queries, page_nums_sample_obj, domains_sample_obj, max_page_num)

    m1, m2 = Manager(), Manager()
    urls_queue, out_queue = m1.Queue(), m2.Queue()

    for q_url in q_urls:
        urls_queue.put(q_url)

    bfs(urls_queue, out_queue, proxies_list_sample_obj, ua, max_level=2, max_threads=max_threads)

    while out_queue.empty() is not True:
        queue_top = out_queue.get()

        search_queries.append(queue_top[0])
        urls.append(queue_top[1])
        url_hashes.append(queue_top[2])
        metadata.append(json.dumps(queue_top[3]))
        timestamp.append(queue_top[4])

    df = pd.DataFrame({'Queries': search_queries, 'URLs': urls, 'URL Hashes': url_hashes, 'Metadata': metadata,
                       'Timestamp': timestamp})

    df.to_csv('amazon_title_desc.csv', index=False, encoding='utf-8')