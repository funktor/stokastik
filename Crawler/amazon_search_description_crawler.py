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
import redis
import datetime
from cassandra.cluster import Cluster
from ssl import SSLContext, PROTOCOL_TLSv1, CERT_REQUIRED
from cassandra.auth import PlainTextAuthProvider
from cassandra import ConsistencyLevel
import constants as cnt


logging.basicConfig(filename=cnt.AMZN_LOGGER, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def insert_metadata(rdis, soup, query, q_url, url_hash, level, out_queue, session, insert_stmt_search, insert_stmt_details):
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

                if metadata['title'] != '':
                    session.execute_async(insert_stmt_search,
                                          [q_url, str(url_hash), query, json.dumps(metadata),
                                           int(datetime.datetime.now().timestamp() * 1000)])
                else:
                    rdis.srem(cnt.AMZN_URL_SET, url_hash)

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

                if metadata['title'] != '':
                    session.execute_async(insert_stmt_search,
                                          [q_url, str(url_hash), query, json.dumps(metadata),
                                           int(datetime.datetime.now().timestamp() * 1000)])
                else:
                    rdis.srem(cnt.AMZN_URL_SET, url_hash)


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

        if metadata['title'] != '':
            session.execute_async(insert_stmt_details,
                                  [q_url, str(url_hash), query, json.dumps(metadata),
                                   int(datetime.datetime.now().timestamp() * 1000)])
        else:
            rdis.srem(cnt.AMZN_URL_SET, url_hash)

    return next_level_urls


def add_to_url_queue(rdis, out_queue, session, insert_stmt_search, insert_stmt_details,
                     throttle, proxies_list_sample_obj, ua):

    while True:
        try:
            out = rdis.blpop(cnt.ELASTICACHE_QUEUE_KEY_AMZN, cnt.REDIS_BLOCKING_TIMEOUT)

            if out is None:
                break

            _, task = out

            x = json.loads(task)
            query, q_url, level = x['query'], x['url'], int(x['level'])

            url_hash = hash(q_url)

            rdis.sadd(cnt.AMZN_URL_SET, url_hash)

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

                next_level_urls = insert_metadata(rdis, soup, query, q_url, url_hash, level, out_queue,
                                                  session, insert_stmt_search, insert_stmt_details)

                if len(next_level_urls) > 0:
                    with rdis.pipeline() as pipe:
                        for url in next_level_urls:
                            if rdis.sismember(cnt.AMZN_URL_SET, hash(url)) == 0:
                                rdis.rpush(cnt.ELASTICACHE_QUEUE_KEY_AMZN,
                                           json.dumps({'query': query, 'url': url, 'level': level + 1}))
                        pipe.execute()

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
    max_threads = cnt.NUM_THREADS
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

    m = Manager()
    out_queue = m.Queue()

    r = redis.StrictRedis(host=cnt.ELASTICACHE_URL, port=cnt.ELASTICACHE_PORT, db=0)

    with r.pipeline() as pipe:
        for q_url in q_urls:
            if r.sismember(cnt.AMZN_URL_SET, hash(q_url[1])) == 0:
                r.rpush(cnt.ELASTICACHE_QUEUE_KEY_AMZN, json.dumps({'query': q_url[0],
                                                                    'url': q_url[1], 'level': q_url[2]}))
        pipe.execute()

    ssl_context = SSLContext(PROTOCOL_TLSv1)
    ssl_context.load_verify_locations(cnt.AWS_KEYSPACES_PEM)
    ssl_context.verify_mode = CERT_REQUIRED
    auth_provider = PlainTextAuthProvider(username=cnt.AWS_KEYSPACES_USER,
                                          password=cnt.AWS_KEYSPACES_PASSWD)

    cluster = Cluster([cnt.CASSANDRA_URL], ssl_context=ssl_context, auth_provider=auth_provider,
                      port=cnt.CASSANDRA_PORT)

    session = cluster.connect(cnt.AMZN_KEYSPACE_NAME)

    session.execute(cnt.AMZN_CREATE_TABLE_SQL_SEARCH)
    insert_stmt_search = session.prepare(cnt.AMZN_INSERT_PREP_STMT_SEARCH)
    insert_stmt_search.consistency_level = ConsistencyLevel.LOCAL_QUORUM

    session.execute(cnt.AMZN_CREATE_TABLE_SQL_DETAILS)
    insert_stmt_details = session.prepare(cnt.AMZN_INSERT_PREP_STMT_DETAILS)
    insert_stmt_details.consistency_level = ConsistencyLevel.LOCAL_QUORUM

    throttle = utils.Throttle(cnt.THROTTLE_TIME)

    threads = [None] * max_threads

    for i in range(max_threads):
        print("Starting thread = ", i)
        threads[i] = threading.Thread(target=add_to_url_queue, args=(r, out_queue, session,
                                                                     insert_stmt_search, insert_stmt_details, throttle,
                                                                     proxies_list_sample_obj, ua))
        threads[i].start()

    print()

    for i in range(max_threads):
        if threads[i]:
            threads[i].join()
            print("Completed thread = ", i)

    r.close()
    session.shutdown()

    while out_queue.empty() is not True:
        queue_top = out_queue.get()

        search_queries.append(queue_top[0])
        urls.append(queue_top[1])
        url_hashes.append(queue_top[2])
        metadata.append(json.dumps(queue_top[3]))
        timestamp.append(queue_top[4])

    df = pd.DataFrame({'Queries': search_queries, 'URLs': urls, 'URL Hashes': url_hashes, 'Metadata': metadata,
                       'Timestamp': timestamp})

    df.to_csv(cnt.AMZN_OUT_FILE, index=False, encoding='utf-8')