import sys, os, importlib
if "/data" in sys.path:
    sys.path.remove("/data")
sys.path.append("/home/jupyter/stormbreaker/deep_learning_models")

os.environ['BASE_PATH']="/home/jupyter/stormbreaker/deep_learning_models"
os.environ['ATTRIBUTE']="furniture_type"

import os, numpy as np, pandas as pd
utils = importlib.import_module('utilities.attribute_extraction.' + os.environ['ATTRIBUTE'] + '.utilities')
import shared_utilities as shutils
from networks.attribute_extraction.network_tf import AttributeExtractionNetwork
import stream_data_generators.attribute_extraction.generator as dg
import constants.attribute_extraction.constants as cnt

# print("Reading input...")
# utils.read_input_file()

# print("Downloading images...")
# utils.download_images()

# print("Creating image data...")
# utils.create_image_data()

# print("Creating text data...")
# utils.create_text_data()

# print("Creating train test...")
# utils.create_train_test()

n = len(shutils.load_data_pkl(cnt.TRAIN_INDICES_PATH))
m = len(shutils.load_data_pkl(cnt.TEST_INDICES_PATH))

transf_labels = shutils.load_data_pkl(cnt.TRANSFORMED_LABELS_PATH)
num_classes = transf_labels.shape[1]

vocab_size = shutils.load_data_pkl(cnt.VOCAB_SIZE_PATH)

print(n, m)

# print("Training model...")
# network = AttributeExtractionNetwork(dg.get_data_as_generator, n, m, num_classes, vocab_size)
# network.fit()

print("Scoring model...")
network = AttributeExtractionNetwork(dg.get_data_as_generator, n, m, num_classes, vocab_size)
network.scoring()
# network.scoring(type='pt')
# network.scoring(type='color')