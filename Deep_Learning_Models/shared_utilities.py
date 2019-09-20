import gensim, tables
from sklearn.model_selection import train_test_split
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from collections import defaultdict
from gensim.models import FastText
import hashlib, itertools, json, pickle, cv2, math, io, time, sys, requests, re
from urllib.parse import urlparse, urlunparse, urljoin, urlencode
import pandas as pd, functools, urllib, urllib.request, urllib.error
import collections, os, random, numpy as np, glob, random, itertools
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from PIL import Image, ImageFile
import http.client, logging
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from io import BytesIO
import shared_constants as cnt
import tensorflow as tf

os.environ['http_proxy'] = "http://sysproxy.wal-mart.com:8080"
os.environ['https_proxy'] = "http://sysproxy.wal-mart.com:8080"

ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_data_npy(data, path):
    np.save(path, data)
    
def load_data_npy(path):
    return np.load(path)
    
def save_data_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
        
def load_data_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def padd_fn(tokens, pad_txt='PAD_TXT', max_len=100):
    return tokens + [pad_txt]*(max_len - len(tokens)) if len(tokens) < max_len else tokens[:max_len]

def clean_tokens(tokens, to_replace='[^\w\-\+\&\.\'\"\:]+'):
    lemma = WordNetLemmatizer()
    
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    tokens = [lemma.lemmatize(token) for token in tokens]
    
    return tokens

def tokenize(mystr):
    tokenizer = RegexpTokenizer('[^ ]+')
    mystr = mystr.lower()

    return tokenizer.tokenize(mystr)

def get_tokens(sentence, to_replace='[^\w\-\+\&\.\'\"\:]+', char_tokens=False):
    sentence = re.sub('<[^<]+?>', ' ', str(sentence))
    sentence = re.sub(to_replace, ' ', str(sentence))
    
    tokens = clean_tokens(tokenize(sentence), to_replace)
    tokens = [token.strip() for token in tokens]
    
    if char_tokens:
        c_tokens = []
        for token in tokens:
            c_tokens += list(token)
        return c_tokens
    
    return tokens

def get_num_batches(num_data, batch_size):
    return int(math.ceil(float(num_data)/batch_size))

def train_wv_model(inp_data, tokenizer_fn, model_path, char_tokens=False):
    tokens = tokenizer_fn(inp_data, char_tokens)
    model = Word2Vec(alpha=0.025, size=128, window=5, min_alpha=0.025, min_count=2, workers=5, negative=10, hs=0, iter=50)

    model.build_vocab(tokens)
    model.train(tokens, total_examples=model.corpus_count, epochs=50)
    model.save(model_path)
    
    
def train_fasttext_model(inp_data, tokenizer_fn, model_path, char_tokens=False):
    tokens = tokenizer_fn(inp_data, char_tokens)
    model = FastText(size=128, window=5, min_count=2, workers=5, iter=50)

    model.build_vocab(tokens)
    model.train(tokens, total_examples=model.corpus_count, epochs=50)
    model.save(model_path)
    
    
def get_vectors(vector_model, tokens, vector_dim):
    output = []
    for tokens in tokens:
        vectors = []
        for word in tokens:
            if word in vector_model.wv:
                vectors.append(vector_model.wv[word])
            else:
                vectors.append([0] * vector_dim)
        output.append(vectors)
    return np.array(output)


def is_walmart_url(host):
    for b in cnt.WMT_HOSTS:
        if b == host:
            return True
    return False


def is_hayneedle_url(host):
    if host == cnt.HYNDL_HOST:
        return True
    return False


def normalize_url(url):
    url = url.strip()
    parsed_url = urlparse(url)
    host = parsed_url.netloc

    if is_hayneedle_url(host):
        url =  urljoin(url, urlparse(url).path)
        return url + '?' + urlencode(cnt.HYNDL_PAYLOAD)

    if is_walmart_url(host):
        return urlunparse(
            ["https", cnt.WMT_MAIN_HOST, parsed_url.path, '', urlencode(cnt.WMT_PAYLOAD), ''])

    return url


def get_base_url(url):
    parsed_url = urlparse(url)
    return urlunparse([parsed_url[0], parsed_url[1], parsed_url[2], '', '', ''])


def url_to_filename(url):
    b_url = get_base_url(url)
    md5_hash = hashlib.md5(b_url.encode('utf-8')).hexdigest()
    return md5_hash + ".jpg"

def process_image(image, image_size):
    try:
        image = Image.open(BytesIO(image)).resize((image_size, image_size), Image.ANTIALIAS).convert('RGB')
    except ValueError as err:
        logging.error("ValueError: " + err.message)
        return -1
    except IOError:
        logging.error("Not a valid image")
        logging.error(url)
        return -2
    
    return image


def image_url_to_obj(url, image_size):
    url = normalize_url(url)
    
    try:
        response = requests.get(url)
        image = process_image(response.content, image_size)
        
    except ValueError as err:
        logging.error("ValueError: " + err.message)
        return -1
    except IOError:
        logging.error("Not a valid image")
        logging.error(url)
        return -2
    
    return image


def image_url_to_array(url, image_size):
    url = normalize_url(url)
    
    try:
        image = np.asarray(image_url_to_obj(url, image_size), dtype='float32')/255.0
        
    except urllib.error.HTTPError as err:
        return err.code
    except urllib.error.URLError as err:
        logging.error("URLError:")
        logging.error(err.reason)
        return -1
    except ValueError as err:
        logging.error("ValueError: " + err.message)
        return -1
    except http.client.BadStatusLine:
        logging.error("Bad Status. Please Investigate")
        return -1
    except IOError:
        logging.error("Not a valid image")
        logging.error(url)
        return -2
    
    return image


def image_to_array(img):
    try:
        image = np.asarray(img, dtype='float32')/255.0
    
    except ValueError as err:
        logging.error("ValueError: " + err.message)
        return -1
    except IOError:
        logging.error("Not a valid image")
        logging.error(url)
        return -2
    
    return image


def fetch_url(url, out_dir):
    url = normalize_url(url)
    try:
        filename = url_to_filename(url)
        
        actualfile = os.path.basename(filename)
        tempstr = filename + "temp"
        temp_out_file = os.path.join(out_dir, tempstr)
        out_file = os.path.join(out_dir, filename)

        if os.path.exists(out_file):
            return 0

        r = urllib.request.urlopen(url)

        img = r.read()
        pim = Image.open(io.BytesIO(img))
        if pim.format != 'JPEG' and pim.format != 'PNG':
            logging.debug("Not a JPEG or PNG image")
            logging.debug(url)
            return -3

        try:
            pim.load()
        except Exception as e:
            logging.error("Image load failed - invalid or incomplete: {}".format(e))
            logging.error(url)
            return -4

        dirname = os.path.dirname(out_file)
        if img and not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != 17:
                    raise

        with open(temp_out_file, 'wb') as f:
            f.write(img)
            f.flush()
            os.fsync(f.fileno())
        os.rename(temp_out_file, out_file)

    except urllib.error.HTTPError as err:
        return err.code
    except urllib.error.URLError as err:
        logging.error("URLError:")
        logging.error(err.reason)
        return -1
    except ValueError as err:
        logging.error("ValueError: " + err.message)
        return -1
    except http.client.BadStatusLine:
        logging.error("Bad Status. Please Investigate")
        return -1
    except IOError:
        logging.error("Not an valid image")
        logging.error(url)
        return -2

    return 200

def fetch_urls_parallel(url_file, out_dir, num_threads):
    urls = []
    batch_size = 1000
    redir_thresh = 100

    num_urls = 0
    results = []
    with open(url_file) as f:
        for line in f:
            num_urls += 1
            urls.append(line.strip())

            if len(urls) == batch_size:
                start_time = time.time()

                pool = ThreadPool(num_threads)
                http_codes = pool.map(lambda u: fetch_url(u, out_dir), urls)
                pool.close()
                pool.join()

                del urls
                urls = []
                results.extend(http_codes)

                batch_redir_cnt = 0
                for http_code in http_codes:
                    if int(http_code)/100 == 3:
                        batch_redir_cnt += 1
                if batch_redir_cnt > redir_thresh or batch_redir_cnt == batch_size:
                    print("Large number of redirections!")
                    break

                ups = 1.0 * batch_size / (time.time() - start_time)
                print("Urls fetched %d. Urls per second %.2f" % (num_urls, ups))

    if len(urls) > 0:
        pool = ThreadPool(num_threads)
        http_codes = pool.map(lambda u: fetch_url(u, out_dir), urls)
        pool.close()
        pool.join()
        results.extend(http_codes)

    print("Total %d urls fetched" % num_urls)
    counter = collections.Counter(results)
    print (dict(counter))


def get_image_array_data(images_path, image_size, num_samples=3000):
    images = glob.glob(os.path.join(images_path, "*.*"))
    images = random.sample(images, min(num_samples, len(images)))
    
    return np.array([img_to_array(load_img(image).resize((image_size, image_size))) for image in images])/255.0

def sort_key(key):
    return -key[1]

def get_features_mi(sentences, labels, num_feats=100, is_multi_label=True):
    a, b, c = defaultdict(float), defaultdict(float), defaultdict(float)

    total = 0

    for idx in range(len(sentences)):
        sent = sentences[idx]
        label = labels[idx]
        
        common_tokens = set(get_tokens(sent))
        
        total += 1
        
        for token in common_tokens:
            a[token] += 1
            
        if is_multi_label:
            for color in label:
                b[color] += 1
        else:
            b[label] += 1
            
        for token in common_tokens:
            if is_multi_label:
                for color in label:
                    c[(color, token)] += 1
            else:
                c[(label, token)] += 1
    
    mi_values = defaultdict(float)

    for key, val in c.items():
        color, token = key

        x11 = val
        x10 = b[color] - val
        x01 = a[token] - val
        x00 = total - (x11 + x10 + x01)

        x1, x0 = b[color], total - b[color]
        y1, y0 = a[token], total - a[token]

        p = float(x11)/total
        q = float(x10)/total
        r = float(x01)/total
        s = float(x00)/total

        u = float(x1)/total
        v = float(x0)/total
        w = float(y1)/total
        z = float(y0)/total

        a1 = p*np.log2(float(p)/(u*w)) if p != 0 else 0
        a2 = q*np.log2(float(q)/(u*z)) if q != 0 else 0
        a3 = r*np.log2(float(r)/(v*w)) if r != 0 else 0
        a4 = s*np.log2(float(s)/(v*z)) if s != 0 else 0

        mi = a1 + a2 + a3 + a4
        
        mi_values[token] = max(mi_values[token], mi)
    
    final_tokens = [(token, val) for token, val in mi_values.items()]
    final_tokens = sorted(final_tokens, key=sort_key)[:min(num_feats, len(final_tokens))]
    
    final_tokens = [x for x, y in final_tokens]
    
    return final_tokens


def get_preprocessed_data(sentences, feature_set=None, tokenizer=None, max_length=200):
    p_sents = []
    for sent in sentences:
        tokens = get_tokens(sent)
        if feature_set is not None:
            tokens = [token for token in tokens if token in feature_set]
        p_sents += [' '.join(tokens)]
    
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(p_sents)

    tensor = tokenizer.texts_to_sequences(p_sents)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length, padding='post')
    
    return tensor, tokenizer
