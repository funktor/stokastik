import os, logging
import shutil
import time
import itertools
import gzip
import argparse
import multiprocessing

import requests
from io import BytesIO

from PIL import Image
import numpy as np
import imagehash
import cv2
import glob
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img

VERY_HIGH_DISTANCE = 1000.00
#NUM_PROCESSES = 16
#PLACEHOLDERS_LOCATION = "./placeholder_examples"
#global_hash_list, global_hist_list = create_signatures(PLACEHOLDERS_LOCATION)


def create_image_hashes(tagged_placeholder_dir):
    placeholder_hashes = set()
    placeholder_image_files = glob.glob("tagged_placeholder_images/*.*")
    
    for img_file in placeholder_image_files:
        img = load_img(img)
        im_hash = imagehash.phash(img)
        placeholder_hashes.add(im_hash)
        
    return placeholder_hashes

def compute_color_historgram(im):
    if len(im.shape) == 2:
        hist_r = np.histogram(im[:, :], bins=256, normed=True)
        hist_g = hist_r
        hist_b = hist_r
    else:
        hist_r = np.histogram(im[:, :, 0], bins=256, normed=True)
        hist_g = np.histogram(im[:, :, 1], bins=256, normed=True)
        hist_b = np.histogram(im[:, :, 2], bins=256, normed=True)
        
    return np.array(list(itertools.chain(hist_r[0], hist_g[0], hist_b[0])))


def create_image_histograms(tagged_placeholder_dir):
    placeholder_histogram = set()
    placeholder_image_files = glob.glob("tagged_placeholder_images/*.*")
    
    for img_file in placeholder_image_files:
        img = load_img(img)
        im_hist = compute_color_historgram(np.asarray(img))
        placeholder_histogram.add(im_hist)
        
    return placeholder_histogram

def create_signatures(tagged_placeholder_dir):
    placeholder_hashes = create_image_hashes(tagged_placeholder_dir)
    placeholder_histogram = create_image_histograms(tagged_placeholder_dir)
    
    return placeholder_hashes, placeholder_histogram



NUM_PROCESSES = 16
PLACEHOLDERS_LOCATION = "/home/achaudhuri/qarth-image-selection/image_selection/placeholder_detection/placeholder_examples"


def compare_against_templates(test_image, test_image_source, hash_list, histo_list):
    try:
        test_im_hash = imagehash.phash(test_image)
    except:
        print "could not create hash of {}".format(test_image_source)
        return VERY_HIGH_DISTANCE, VERY_HIGH_DISTANCE

    try:
        test_im_hist = compute_color_historgram(np.asarray(test_image))
    except:
        print "could not create histogram of {}".format(test_image_source)
        return VERY_HIGH_DISTANCE, VERY_HIGH_DISTANCE

    min_hash_dist = 100
    for h in hash_list:
        hash_dist = abs(h - test_im_hash)
        if hash_dist < min_hash_dist:
            min_hash_dist = hash_dist

    min_histo_dist = 100
    n_bins = 3 * 256
    for h in histo_list:
        histo_dist = cv2.compareHist(np.float32(test_im_hist), np.float32(h), cv2.HISTCMP_CHISQR) / n_bins
        if histo_dist < min_histo_dist:
            min_histo_dist = histo_dist

    return min_hash_dist, min_histo_dist


def is_placeholder(test_image_source, hash_list, histo_list, hash_threshold=2, histo_threshold=0.001, is_url=False):
    try:
        if is_url is True:
            response = requests.get(test_image_source)
            test_image = Image.open(BytesIO(response.content)).resize((128, 128), Image.ANTIALIAS)
        else:
            test_image = Image.open(test_image_source).resize((128, 128), Image.ANTIALIAS)
    except Exception as e:
        print "could not load or resize {0}, message: {1}".format(test_image_source, e.message)
        return False, None, None, None

    hash_dist, histo_dist = compare_against_templates(test_image, test_image_source, hash_list, histo_list)
    if hash_dist <= hash_threshold and histo_dist <= histo_threshold:
        return True, hash_dist, histo_dist, test_image
    return False, hash_dist, histo_dist, test_image


def is_placeholder_image(test_image, hash_list, histo_list, hash_threshold=2, histo_threshold=0.001):
    try:
        test_image = test_image.resize((128, 128), Image.ANTIALIAS)
    except:
        print "could not resize test image"
        return False

    hash_dist, histo_dist = compare_against_templates(test_image, None, hash_list, histo_list)
    if hash_dist <= hash_threshold and histo_dist <= histo_threshold:
        return True, hash_dist, histo_dist, test_image
    return False, hash_dist, histo_dist, test_image


def is_placeholder_image_for_service(test_image, hash_list, histo_list, hash_threshold=2, histo_threshold=0.002):
    img = Image.open(BytesIO(test_image))
    img = img.resize((128, 128), Image.ANTIALIAS)
    hash_dist, histo_dist = compare_against_templates(img, None, hash_list, histo_list)
    if hash_dist <= hash_threshold and histo_dist <= histo_threshold:
        hash_confidence = 1.0 - float(hash_dist)/float(hash_threshold)
        histo_confidence = 1.0 - float(histo_dist) / float(histo_threshold)
        placeholder_confidence = 0.75 * hash_confidence + 0.2 * histo_confidence
        return True, placeholder_confidence

    hash_confidence = np.clip(float(hash_dist - hash_threshold) / float(hash_threshold), 0.0, 1.0)
    histo_confidence = np.clip(float(histo_dist - histo_threshold) / float(histo_threshold), 0.0, 1.0)
    not_placeholder_confidence = 0.75 * hash_confidence + 0.2 * histo_confidence
    return False, not_placeholder_confidence

def is_placeholder_worker(test_image_source):
    hash_threshold = 0
    histo_threshold = 0.001
    try:
        response = requests.get(test_image_source[1])
        test_image = Image.open(BytesIO(response.content)).resize((128, 128), Image.ANTIALIAS)
    except:
        print "could not load or resize {}".format(test_image_source[1])
        return

    hash_dist, histo_dist = compare_against_templates(test_image, test_image_source[1], global_hash_list, global_hist_list)
    if hash_dist <= hash_threshold and histo_dist <= histo_threshold:
        with open('/tmp/detected_secondary_placeholders_dec6.txt', 'a') as outfile:
            outfile.write("{0}\t{1}\n".format(test_image_source[0], test_image_source[1]))


def run_algo_on_benchmark():
    IMG_FOLDER = "/Users/achaud9/Box Sync/Data/ImageScoring (achaudhuri@walmartlabs.com)"
    IMG_DATASET_FOLDER = "{}/benchmark_placeholder".format(IMG_FOLDER)
    INDEX_FILE = "{}/index.txt".format(IMG_DATASET_FOLDER)
    IMAGES_DIR = "{}/images".format(IMG_DATASET_FOLDER)
    RESULTS_FOLDER = "{}/results".format(IMG_FOLDER)
    RESULTS_FILE = "{}/placeholder_detect_summary.txt".format(RESULTS_FOLDER)
    PLACEHOLDERS_LOCATION = "./placeholder_examples"

    ph_hash_list, ph_hist_list = create_signatures(PLACEHOLDERS_LOCATION)

    true_labels = []
    predicted_labels = []

    start_time = time.time()

    with open(INDEX_FILE, 'rb') as infile, open(RESULTS_FILE, 'wb') as outfile:
        index = 0
        for line in infile:
            fields = line.rstrip('\n').split()
            # print fields

            image_relative_path = fields[0]
            true_label = int(fields[1])
            image_full_path = os.path.join(IMG_DATASET_FOLDER, image_relative_path)

            try:
                im = Image.open(image_full_path)
            except IOError:
                print "could not load {}".format(image_full_path)
                index += 1
                continue

            decision, hash_dist, histo_dist = is_placeholder(image_full_path, ph_hash_list, ph_hist_list,
                                                             hash_threshold=2, histo_threshold=0.001)

            if decision is True:
                prediction = 1
            else:
                prediction = 0

            outfile.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\n".format(image_relative_path, hash_dist, histo_dist, true_label, prediction))

            if true_label != prediction:
                print "detection error at {}".format(image_relative_path)
                print "{0}\t{1}\t{2}\t{3}\t{4}\n".format(image_relative_path, hash_dist, histo_dist, true_label,
                                                         prediction)

            true_labels.append(true_label)
            predicted_labels.append(prediction)

            index += 1
            if index % 50 == 0:
                print "{} done".format(index)

    end_time = time.time()
    print "Time taken to process {0} images: {1}".format(index, (end_time - start_time))

    a = accuracy_score(true_labels, predicted_labels)
    p = precision_score(true_labels, predicted_labels)
    r = recall_score(true_labels, predicted_labels)
    f = f1_score(true_labels, predicted_labels)

    print "Threshold:{0},{1} Metrics:{2}\t{3}\t{4}\t{5}".format(2, 0.001, a, p, r, f)


def extract_urls_from_catalog_dump(dump_file, primary_urls_file, secondary_urls_file):
    with open(dump_file, 'rb') as infile, \
            open(primary_urls_file, 'wb') as outfile1, \
            open(secondary_urls_file, 'wb') as outfile2:

        url_list = []
        data_count = 0
        for line in infile:
            fields = line.rstrip('\n').split('\t')
            # print fields

            if len(fields) != 3:
                print "corrupt line: {}".format(fields)
                continue

            wpid = fields[0]
            urls = fields[1].lstrip('[').rstrip(']').split(",")
            im_types = fields[2].lstrip('[').rstrip(']').split(",")
            urls = [x.strip("\"") for x in urls]
            im_types = [x.strip("\"") for x in im_types]

            if len(urls) != len(im_types):
                print "image urls do not have types"
                continue

            for i, v in enumerate(urls):

                im_url = v.strip('\"')
                #print im_url
                if im_url == 'NULL':
                    continue

                if im_types[i].strip("\"") == 'PRIMARY':
                    outfile1.write("{0}\t{1}\n".format(wpid, im_url))
                elif im_types[i].strip("\"") == 'SECONDARY':
                    outfile2.write("{0}\t{1}\n".format(wpid, im_url))
                else:
                    continue



def run_algo_on_catalog_dump(IMG_URLS_FILE):
    #PLACEHOLDERS_LOCATION = "./placeholder_examples"
    #ph_hash_list, ph_hist_list = create_signatures(PLACEHOLDERS_LOCATION)

    start_time = time.time()
    loop_start_time = start_time

    with gzip.open(IMG_URLS_FILE, 'rb') as infile:
        index = 0
        global_im_index = 0
        url_list = []
        data_count = 0
        for line in infile:
            fields = line.rstrip('\n').split('\t')
            # print fields

            if len(fields) != 3:
                print "corrupt line: {}".format(fields)
                continue

            pt = fields[0]
            wpid = fields[1]
            urls = fields[2].lstrip('[').rstrip(']').split(",")
            # print urls

            local_im_index = 0
            for l in urls:

                im_url = l.strip('\"')
                #print im_url
                if im_url == 'NULL':
                    continue

                url_list.append((wpid, im_url))
                data_count += 1
                #print data_count, NUM_PROCESSES
                #if data_count == NUM_PROCESSES:
                    #print "about to launch parallel job"
		            #print url_list
		            #jobs = list()
                    #for i,v in enumerate(url_list):
                    #    p = multiprocessing.Process(target=is_placeholder_worker, args=(v,))
                    #jobs.append(p)
                    #p.start()

                    #global_im_index += data_count
                    #data_count = 0
                    #url_list = []

            index += 1
            if index % 100 == 0:
                print "{} done".format(index)
                print "{} images checked".format(global_im_index)
                end_time = time.time()
                print "Time taken to process 100 WPIDS: {0} s, total: {1}".format((end_time - loop_start_time), (end_time - start_time))
                loop_start_time = end_time

    end_time = time.time()
    print "Time taken to process {0} WPIDS with {1} images: {2}".format(index, global_im_index, (end_time - start_time))


def run_algo_on_image_list(IMG_URLS_FILE):

    start_time = time.time()
    loop_start_time = start_time

    with open(IMG_URLS_FILE, 'rb') as infile:
        index = 0
        global_im_index = 0
        url_list = []
        data_count = 0
        for line in infile:
            fields = line.rstrip('\n').split('\t')

            if len(fields) != 2:
                print "corrupt line: {}".format(fields)
                continue

            wpid = fields[0]
            im_url = fields[1]
            if im_url == 'NULL':
                continue

            url_list.append((wpid, im_url))
            data_count += 1

            if data_count == NUM_PROCESSES:
                jobs = []
                for i,v in enumerate(url_list):
                    p = multiprocessing.Process(target=is_placeholder_worker, args=(v,))
                    jobs.append(p)
                    p.start()

                data_count = 0
                url_list = []

            index += 1
            if index % 100 == 0:
                print "{} done".format(index)
                end_time = time.time()
                print "Time taken to process 100 WPIDS: {0} s, total: {1}".format((end_time - loop_start_time), (end_time - start_time))
                loop_start_time = end_time

    end_time = time.time()
    print "Time taken to process {0} WPIDS with {1} images: {2}".format(index, global_im_index, (end_time - start_time))


def run_algo_on_image_dir(image_dir, template_hash_list, template_histo_list,
                          hash_threshold = 0, histo_threshold = 0.001,
                          move_to_dir='/tmp'):

    start_time = time.time()
    loop_start_time = start_time

    index = 0
    for filename in os.listdir(image_dir):
        if filename.startswith("."):
            continue

        image_path = os.path.join(image_dir, filename)

        flag, hash_dist, histo_dist, test_image =  is_placeholder(image_path, template_hash_list, template_histo_list,
                                                                hash_threshold=hash_threshold, histo_threshold=histo_threshold)

        if flag is True:
            dest_path = os.path.join(move_to_dir, filename)
            shutil.move(image_path, dest_path)

        index += 1
        if index % 100 == 0:
            print "{} done".format(index)
            end_time = time.time()
            print "Time taken to process 100 images: {0} s, total: {1}".format((end_time - loop_start_time), (end_time - start_time))
            loop_start_time = end_time

    end_time = time.time()
    print "Time taken to process {0} WPIDS with {1} images: {2}".format(index, index, (end_time - start_time))


def create_dataset():
    IMG_FOLDER = "/Users/achaud9/Box Sync/Data/ImageScoring (achaudhuri@walmartlabs.com)"
    IMG_DATASET_FOLDER = "{}/benchmark_placeholder".format(IMG_FOLDER)
    INDEX_FILE = "{}/index.txt".format(IMG_DATASET_FOLDER)
    IMAGES_DIR = "{}/images".format(IMG_DATASET_FOLDER)
    nextid = 0
    with open(INDEX_FILE, 'wb') as output_file:
        for f in os.listdir(IMAGES_DIR):
            if "_ph" in f:
                output_file.write("images/{0}\t1\n".format(f))
            elif len(f) > 20:
                extn = f.split(".")[-1]
                new_name = "fromcatalog_{0}.{1}".format(nextid, extn)
                os.rename(os.path.join(IMAGES_DIR, f), os.path.join(IMAGES_DIR, new_name))
                output_file.write("images/{0}\t0\n".format(new_name))
                nextid += 1
            else:
                output_file.write("images/{0}\t0\n".format(f))


def test_is_placeholder():

    test_image_source = "https://i5.walmartimages.com/asr/67c27861-7f84-43f7-ab25-e5b36c414944_1.a2349b6b23a5837b69dccb3e615781ad.jpeg"
    flag, d1, d2, im = is_placeholder(test_image_source, global_hash_list, global_hist_list, hash_threshold=0, histo_threshold=0.001, is_url=True)
    print flag
    print d1, d2


if __name__ == "__main__":
    global_hash_list, global_hist_list = create_signatures(PLACEHOLDERS_LOCATION)

    # create_dataset()
    # run_algo_on_benchmark()


    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help='The file with the image urls', action="store", default=None)
    parser.add_argument("--n_cores", help='The number od CPU cores to use', action="store", default=None)
    #parser.add_argument("--out_file", help='The file where results are written', action="store", default=None)
    #parser.add_argument("--out_folder", help='The folder where placeholders are written as small icons', action="store",
    #                    default=None)

    args = parser.parse_args()
    #for i in range(0,10):
    #	run_algo_on_catalog_dump('/home/achaudhuri/data/image_scoring/placeholder_detection/dump{}.txt.gz'.format(i+1))
    #print "part {} done".format(i)

    run_algo_on_image_list(args.input_file)

    #run_algo_on_image_dir("/Users/achaud9/Work/Data/ImageClassification/TestImages",
                          #global_hash_list, global_hist_list,
                         #hash_threshold=0, histo_threshold=0.001,
                         #move_to_dir="/Users/achaud9/Work/Data/ImageClassification/detected_placeholders")

    #POC_DIR = "/Users/achaud9/Work/Data/ImageScoring (achaudhuri@walmartlabs.com)/poc_placeholder/results/oct25"
    #extract_urls_from_catalog_dump("{}/images_for_placeholder_set_ids.txt".format(POC_DIR),
    #                               "{}/primary_images_to_remove.txt".format(POC_DIR),
    #                               "{}/secondary_images_to_remove.txt".format(POC_DIR))#
