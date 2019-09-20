import os, random, numpy as np, math, tables
import constants.neural_machine_translation.constants as cnt
import shared_utilities as shutils
import utilities.neural_machine_translation.utilities as utils
import tensorflow as tf

def get_data_as_generator(num_samples, type='train'):
    if type == 'train':
        src_tensor_train = shutils.load_data_pkl(cnt.SRC_TENSOR_TRAIN)
        trg_tensor_train = shutils.load_data_pkl(cnt.TRG_TENSOR_TRAIN)
        
        n = len(src_tensor_train)
        
        dataset = tf.data.Dataset.from_tensor_slices((src_tensor_train, trg_tensor_train)).shuffle(n)
        
    else:
        src_tensor_valid = shutils.load_data_pkl(cnt.SRC_TENSOR_VALID)
        trg_tensor_valid = shutils.load_data_pkl(cnt.TRG_TENSOR_VALID)
        
        n = len(src_tensor_valid)
        
        dataset = tf.data.Dataset.from_tensor_slices((src_tensor_valid, trg_tensor_valid)).shuffle(n)
        
    dataset = dataset.batch(cnt.BATCH_SIZE, drop_remainder=False)
    
    return iter(dataset)