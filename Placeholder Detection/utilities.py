import hashlib, itertools, json, pickle, cv2, math, io, time, sys, requests, re
from urllib.parse import urlparse, urlunparse, urljoin, urlencode
import pandas as pd, functools, urllib, urllib.request, urllib.error
import collections, os, random, numpy as np, glob, random, itertools
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.model_selection import train_test_split
import constants as cnt, http.client, logging
from siamese_predictor import SiameseModel
from sklearn.neighbors import KDTree
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from io import BytesIO


def load_siamese_model():
    model = SiameseModel()
    model.init_model()
    model.load()
    
    return model


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


def image_url_to_obj(url):
    url = normalize_url(url)
    
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).resize((cnt.IMAGE_SIZE, cnt.IMAGE_SIZE), Image.ANTIALIAS)
        image = image.convert('RGB')
        
    except ValueError as err:
        logging.error("ValueError: " + err.message)
        return -1
    except IOError:
        logging.error("Not a valid image")
        logging.error(url)
        return -2
    
    return image


def image_url_to_array(url):
    url = normalize_url(url)
    
    try:
        image = np.asarray(image_url_to_obj(url), dtype='float32')/255.0
        
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
        if pim.format != 'JPEG' and pim.format != 'PNG' and pim.format != 'GIF':
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


def get_image_array_data(images_path, num_samples=3000):
    images = glob.glob(os.path.join(images_path, "*.*"))
    images = random.sample(images, min(num_samples, len(images)))
    
    return np.array([img_to_array(load_img(image).resize((cnt.IMAGE_SIZE, cnt.IMAGE_SIZE))) for image in images])/255.0


def save_data_npy(data, path):
    np.save(os.path.join(cnt.DATA_DIR, path), data)
    
    
def load_data_npy(path):
    return np.load(os.path.join(cnt.DATA_DIR, path))

    
def save_data_pkl(data, path):
    with open(os.path.join(cnt.DATA_DIR, path), 'wb') as f:
        pickle.dump(data, f)
        
        
def load_data_pkl(path):
    with open(os.path.join(cnt.DATA_DIR, path), 'rb') as f:
        return pickle.load(f, encoding='latin1')
    
    
def get_siamese_data():
    image_size = cnt.IMAGE_SIZE
    
    product_type_dirs = glob.glob(os.path.join(cnt.PRODUCT_IMAGES_PATH, "*/"))

    train_placeholders = glob.glob(os.path.join(cnt.TAGGED_PLACEHOLDER_IMAGES_PATH, "*.*"))
    test_placeholders = glob.glob(os.path.join(cnt.TEST_PLACEHOLDER_IMAGES_PATH, "*.*"))

    train_urls, valid_urls = [], []

    train_pt_url_map, valid_pt_url_map = collections.defaultdict(list), collections.defaultdict(list)
    train_url_pt_map, valid_url_pt_map = dict(), dict()

    for pt_dir in product_type_dirs:
        curr_urls = glob.glob(os.path.join(pt_dir, "*.*"))
        pt = re.sub('.*\\/(.*?)\\/$', '\\1', pt_dir)

        x, y = train_test_split(curr_urls, test_size=0.2)

        train_urls += zip(x, [pt]*len(x))
        valid_urls += zip(y, [pt]*len(y))

    print(len(train_urls), len(valid_urls))

    train_img_data = np.empty((len(train_urls), image_size, image_size, 3), np.float32)

    for i in range(len(train_urls)):
        img_path, pt = train_urls[i]
        train_pt_url_map[pt].append(i)
        train_url_pt_map[i] = pt
        train_img_data[i] = img_to_array(load_img(img_path).resize((image_size, image_size)))/255.0

    save_data_npy(train_img_data, cnt.TRAIN_IMAGE_DATA_FILE)
    save_data_pkl(train_pt_url_map, cnt.TRAIN_PT_URL_MAP_FILE)
    save_data_pkl(train_url_pt_map, cnt.TRAIN_URL_PT_MAP_FILE)

    valid_img_data = np.empty((len(valid_urls), image_size, image_size, 3), np.float32)

    for i in range(len(valid_urls)):
        img_path, pt = valid_urls[i]
        valid_pt_url_map[pt].append(i)
        valid_url_pt_map[i] = pt
        valid_img_data[i] = img_to_array(load_img(img_path).resize((image_size, image_size)))/255.0

    save_data_npy(valid_img_data, cnt.VALID_IMAGE_DATA_FILE)
    save_data_pkl(valid_pt_url_map, cnt.VALID_PT_URL_MAP_FILE)
    save_data_pkl(valid_url_pt_map, cnt.VALID_URL_PT_MAP_FILE)

    train_placeholders_data = [img_to_array(load_img(img_path).resize((image_size, image_size)))/255.0 for img_path in train_placeholders]
    test_placeholders_data = [img_to_array(load_img(img_path).resize((image_size, image_size)))/255.0 for img_path in test_placeholders]

    save_data_npy(train_placeholders_data, cnt.TRAIN_PLACEHOLDERS_FILE)
    save_data_npy(test_placeholders_data, cnt.TEST_PLACEHOLDERS_FILE)
    
    
def get_sampled_train_data():
    random.seed(42)
    
    train_image_data = load_data_npy(cnt.TRAIN_IMAGE_DATA_FILE)
    train_pt_url_map = load_data_pkl(cnt.TRAIN_PT_URL_MAP_FILE)
    
    sampled, q = [], 0
    for pt, urls in train_pt_url_map.items():
        w = random.sample(urls, min(len(urls), cnt.NUM_SAMPLES_PER_PT))
        train_pt_url_map[pt] = [q + x for x in range(len(w))]
        q += len(train_pt_url_map[pt])
        sampled += w
    
    return train_image_data[sampled], train_pt_url_map

    
def get_sampled_voting_data():
    train_image_data = load_data_npy(cnt.TRAIN_IMAGE_DATA_FILE)
    train_pt_url_map = load_data_pkl(cnt.TRAIN_PT_URL_MAP_FILE)
    
    voting_data = dict()
    
    for pt, urls in train_pt_url_map.items():
        voting_data[pt] = train_image_data[urls]
        
    return voting_data