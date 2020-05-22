import requests, re, urllib
from bs4 import BeautifulSoup
import pandas as pd, numpy as np
import time, math, random
from multiprocessing import Process, Queue, Pool, Manager
import threading
import logging
from fake_useragent import UserAgent
import crawler_utils as utils
import urllib.parse

logging.basicConfig(filename='crawler.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def get_data_search(q_urls, queue, throttle, proxies_list_sample_obj, ua):
    for query, q_url in q_urls:
        try:
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

                for d in soup.findAll('div', attrs={'class': 'sg-col-4-of-12 sg-col-8-of-16 sg-col-16-of-24 sg-col-12-of-20 sg-col-24-of-32 sg-col sg-col-28-of-36 sg-col-20-of-28'}):

                    p_title = d.find('span', attrs={'class': 'a-size-medium a-color-base a-text-normal'})
                    p_price = d.find('span', attrs={'class': 'a-offscreen'})
                    p_rating = d.find('span', attrs={'class': 'a-icon-alt'})
                    p_author = d.find('a', attrs={'class': 'a-size-base a-link-normal'})
                    p_product_url = d.find('a', attrs={'class': 'a-link-normal a-text-normal'}, href=True)

                    all = []

                    if p_title is not None:
                        all.append(utils.sanitize(p_title.text))
                    else:
                        all.append("NA")

                    if p_price is not None:
                        all.append(p_price.text)
                    else:
                        all.append('NA')

                    if p_rating is not None:
                        all.append(p_rating.text)
                    else:
                        all.append('NA')

                    if p_author is not None:
                        all.append(p_author.text)
                    else:
                        all.append('NA')

                    if p_product_url is not None:
                        all.append(p_product_url['href'])
                    else:
                        all.append('NA')

                    all.append(query)

                    queue.put(all)

                for d in soup.findAll('div', attrs={'class': 'sg-col-4-of-24 sg-col-4-of-12 sg-col-4-of-36 s-result-item s-asin sg-col-4-of-28 sg-col-4-of-16 sg-col sg-col-4-of-20 sg-col-4-of-32'}):

                    p_title = d.find('span', attrs={'class': 'a-size-base-plus a-color-base a-text-normal'})
                    p_price = d.find('span', attrs={'class': 'a-offscreen'})
                    p_rating = d.find('span', attrs={'class': 'a-icon-alt'})
                    p_author = d.find('a', attrs={'class': 'a-size-base a-link-normal'})
                    p_product_url = d.find('a', attrs={'class': 'a-link-normal a-text-normal'}, href=True)

                    all = []

                    if p_title is not None:
                        all.append(utils.sanitize(p_title.text))
                    else:
                        all.append("NA")

                    if p_price is not None:
                        all.append(p_price.text)
                    else:
                        all.append('NA')

                    if p_rating is not None:
                        all.append(p_rating.text)
                    else:
                        all.append('NA')

                    if p_author is not None:
                        all.append(p_author.text)
                    else:
                        all.append('NA')

                    if p_product_url is not None:
                        all.append(p_product_url['href'])
                    else:
                        all.append('NA')

                    all.append(query)

                    queue.put(all)

        except Exception as e:
            logger.error(e)
            time.sleep(5.0)


def get_urls(queries, page_nums_sample_obj, domains_sample_obj, max_page_num=5):
    page_nums = page_nums_sample_obj.get(max_page_num)

    urls = []
    for query in queries:
        for page_num in page_nums:
            qs = urllib.parse.urlencode({'k' : query, 'page' : str(page_num)})
            rand_domain = domains_sample_obj.get_one()
            urls.append((query, 'https://' + rand_domain + '/s?' + qs))

    random.shuffle(urls)
    return urls

if __name__ == "__main__":
    max_threads = 50
    max_page_num = 50

    titles, ratings, prices, detail_urls, authors, query_strs = [], [], [], [], [], []

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
        proxies_list_weights = [1.0]*10 + [0.25]*(len(proxies_list)-10)

    proxies_list_sample_obj = utils.Sample(proxies_list, proxies_list_weights)

    page_nums_sample_obj = utils.Sample(list(range(1, 2*max_page_num+1)), [1.0]*max_page_num + [0.2]*max_page_num,
                                        False)

    q_urls = get_urls(queries, page_nums_sample_obj, domains_sample_obj, max_page_num)

    m = Manager()
    queue = m.Queue()

    throttle = utils.Throttle(1.0)

    batch_size = int(math.ceil(len(q_urls)/float(max_threads)))
    threads = [None]*max_threads

    for i in range(max_threads):
        print("Starting thread = ", i)
        start, end = i*batch_size, min((i+1)*batch_size, len(q_urls))
        urls = [q_urls[j] for j in range(start, end)]

        threads[i] = threading.Thread(target=get_data_search, args=(urls, queue, throttle,
                                                                    proxies_list_sample_obj, ua))
        threads[i].start()

    print()

    for i in range(max_threads):
        if threads[i]:
            threads[i].join()
            print("Completed thread = ", i)

    while queue.empty() is not True:
        queue_top = queue.get()
        titles.append(queue_top[0])
        prices.append(queue_top[1])
        ratings.append(queue_top[2])
        detail_urls.append(queue_top[3])
        authors.append(queue_top[4])
        query_strs.append(queue_top[5])

    df = pd.DataFrame({'Query': query_strs, 'Product Name': titles, 'Price': prices, 'Ratings': ratings,
                       'Authors': authors,
                       'Product Details Page': detail_urls})

    df.to_csv('amazon_products_6.csv', index=False, encoding='utf-8')