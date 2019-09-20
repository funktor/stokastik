import os, random, numpy as np, math, tables, importlib
import constants.attribute_extraction.constants as cnt
import shared_utilities as shutils
utils = importlib.import_module('utilities.attribute_extraction.' + os.environ['ATTRIBUTE'] + '.utilities')

def get_data_as_generator(num_data, prefix='train'):
    try:
        img_arr_file = tables.open_file(cnt.IMAGE_ARRAY_PATH, mode='r')
        img_arr = img_arr_file.root.data
        
        txt_arr = shutils.load_data_pkl(cnt.INPUT_TENSOR_PATH)
        txt_arr = np.array(txt_arr)
        
        labels = shutils.load_data_pkl(cnt.TRANSFORMED_LABELS_PATH)
            
        labels = np.array(labels)
        random.seed(42)
        
        if prefix == 'train':
            indices = shutils.load_data_pkl(cnt.TRAIN_INDICES_PATH)
        else:
            indices = shutils.load_data_pkl(cnt.TEST_INDICES_PATH)
        
        random.shuffle(indices)
        indices = np.array(indices)

        num_batches = int(math.ceil(float(num_data)/cnt.BATCH_SIZE))

        batch_num = 0

        while True:
            m = batch_num % num_batches

            start, end = m*cnt.BATCH_SIZE, min((m+1)*cnt.BATCH_SIZE, num_data)
            batch_indices = indices[start:end]
            
            out_img_arr = np.array([img_arr[x] for x in batch_indices])
            out_txt_arr = np.array([txt_arr[x] for x in batch_indices])
            
            batch_num += 1

            yield [out_img_arr, out_txt_arr], labels[batch_indices]
            
    finally:
        img_arr_file.close()
