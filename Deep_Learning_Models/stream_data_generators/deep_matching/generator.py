import os, random, numpy as np, math
import constants.deep_matching.constants as cnt
import utilities.deep_matching.utilities as utils
import shared_utilities as shutils

def get_data_as_generator(num_data, prefix='train'):
    random.seed(42)
    
    word_vector_model = utils.get_vector_model(cnt.VECTOR_MODEL, char_tokens=False)
    char_vector_model = utils.get_vector_model(cnt.VECTOR_MODEL, char_tokens=True)
    
    data_pairs = shutils.load_data_pkl(os.path.join(cnt.PERSISTENCE_PATH, prefix + "_data_pairs.pkl"))
    random.shuffle(data_pairs)

    num_batches = shutils.get_num_batches(num_data, cnt.BATCH_SIZE)

    batch_num = 0

    while True:
        m = batch_num % num_batches

        start, end = m*cnt.BATCH_SIZE, min((m+1)*cnt.BATCH_SIZE, num_data)
        
        word_tokens1, word_tokens2, char_tokens1, char_tokens2, labels = zip(*data_pairs[start:end])
        labels = np.array(labels)
        labels = np.expand_dims(labels, -1)

        word_data_1 = shutils.get_vectors(word_vector_model, word_tokens1, cnt.WORD_VECTOR_DIM)
        word_data_2 = shutils.get_vectors(word_vector_model, word_tokens2, cnt.WORD_VECTOR_DIM)
        
        char_data_1 = np.array([shutils.get_vectors(char_vector_model, x, cnt.CHAR_VECTOR_DIM) for x in char_tokens1])
        char_data_2 = np.array([shutils.get_vectors(char_vector_model, x, cnt.CHAR_VECTOR_DIM) for x in char_tokens2])

        batch_num += 1

        yield [word_data_1, word_data_2, char_data_1, char_data_2], labels