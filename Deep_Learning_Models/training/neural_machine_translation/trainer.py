import sys, os
if "/data" in sys.path:
    sys.path.remove("/data")
sys.path.append("/home/jupyter/stormbreaker/deep_learning_models")
os.environ['BASE_PATH']="/home/jupyter/stormbreaker/deep_learning_models"

import os, numpy as np, pandas as pd
import utilities.neural_machine_translation.utilities as utils
import shared_utilities as shutils
from networks.neural_machine_translation.network_tf import NMTNetwork
import stream_data_generators.neural_machine_translation.generator as dg
import constants.neural_machine_translation.constants as cnt
from sklearn.model_selection import train_test_split

# src_tensor, trg_tensor, src_lang, trg_lang = utils.load_dataset(cnt.DATASET_FILEPATH, cnt.NUM_EXAMPLES)
# max_length_trg, max_length_src = utils.max_length(trg_tensor), utils.max_length(src_tensor)

# src_tensor_train, src_tensor_valid, trg_tensor_train, trg_tensor_valid = train_test_split(src_tensor, trg_tensor, test_size=0.2)

# shutils.save_data_pkl(src_tensor_train, cnt.SRC_TENSOR_TRAIN)
# shutils.save_data_pkl(src_tensor_valid, cnt.SRC_TENSOR_VALID)
# shutils.save_data_pkl(trg_tensor_train, cnt.TRG_TENSOR_TRAIN)
# shutils.save_data_pkl(trg_tensor_valid, cnt.TRG_TENSOR_VALID)

# shutils.save_data_pkl(src_tensor, cnt.SRC_TENSOR)
# shutils.save_data_pkl(trg_tensor, cnt.TRG_TENSOR)

# shutils.save_data_pkl(src_lang, cnt.SRC_LANG)
# shutils.save_data_pkl(trg_lang, cnt.TRG_LANG)

# print(len(src_tensor_train), len(trg_tensor_train), len(src_tensor_valid), len(trg_tensor_valid))


src_tensor_train = shutils.load_data_pkl(cnt.SRC_TENSOR_TRAIN)
trg_tensor_train = shutils.load_data_pkl(cnt.TRG_TENSOR_TRAIN)
src_tensor_valid = shutils.load_data_pkl(cnt.SRC_TENSOR_VALID)
trg_tensor_valid = shutils.load_data_pkl(cnt.TRG_TENSOR_VALID)

src_lang = shutils.load_data_pkl(cnt.SRC_LANG)
trg_lang = shutils.load_data_pkl(cnt.TRG_LANG)

src_tensor = shutils.load_data_pkl(cnt.SRC_TENSOR)
trg_tensor = shutils.load_data_pkl(cnt.TRG_TENSOR)

max_length_src = utils.max_length(src_tensor)
max_length_trg = utils.max_length(trg_tensor)

n, m = len(src_tensor_train), len(src_tensor_valid)

nmt = NMTNetwork(dg.get_data_as_generator, n, m, src_lang, trg_lang, max_length_src, max_length_trg)
# nmt.fit()

test_sentence = "<UL>Full Coverage Cups<LI>Nursing favorite<LI>Superior fit<LI>Lightly padded cups<LI>Great t-shirt bra<LI>All day comfort<LI>Quick fold down cups<LI>Easy one hand nursing clasp&nbsp;<LI>Soft inner sling<LI>Underwire for additional support<LI>Adjustable back closure and straps&nbsp;</UL>"

print(nmt.predict(test_sentence))

