import os, numpy as np, pandas as pd
import utilities.deep_matching.utilities as utils
import shared_utilities as shutils
from networks.deep_matching.network import DeepMatchingNetwork
import constants.deep_matching.constants as cnt

class Inference:
    def __init__(self):
        self.vector_model = utils.get_vector_model(cnt.VECTOR_MODEL)
        self.network = DeepMatchingNetwork()
        self.network.init_model()
        self.network.load()
        

    def predict(self, input_data):
        try:
            title_1, title_2 = input_data['title_1'], input_data['title_2']

            tokens1 = np.array([shutils.padd_fn(shutils.get_tokens(title_1), max_len=cnt.MAX_WORDS)])
            tokens2 = np.array([shutils.padd_fn(shutils.get_tokens(title_2), max_len=cnt.MAX_WORDS)])

            sent_data_1 = shutils.get_vectors(self.vector_model, tokens1, cnt.VECTOR_DIM)
            sent_data_2 = shutils.get_vectors(self.vector_model, tokens2, cnt.VECTOR_DIM)

            prediction, probability = self.network.predict([sent_data_1, sent_data_2], return_probability=True)

            return {"status": 1, "is_match" : int(prediction[0]), "confidence" : float(probability[0])}
        
        except Exception as err:
            return {"status": 0, "message" : str(err)}
    
    
    def predict_batch(self, multi_input_data):
        try:
            tokens1, tokens2 = [], []

            for input_data in multi_input_data:
                title_1, title_2 = input_data['title_1'], input_data['title_2']

                tokens1.append(shutils.padd_fn(shutils.get_tokens(title_1), max_len=cnt.MAX_WORDS))
                tokens2.append(shutils.padd_fn(shutils.get_tokens(title_2), max_len=cnt.MAX_WORDS))

            tokens1 = np.array(tokens1)
            tokens2 = np.array(tokens2)

            sent_data_1 = shutils.get_vectors(self.vector_model, tokens1, cnt.VECTOR_DIM)
            sent_data_2 = shutils.get_vectors(self.vector_model, tokens2, cnt.VECTOR_DIM)

            prediction, probability = self.network.predict([sent_data_1, sent_data_2], return_probability=True)

            return {"status": 1, "response" : [{"status": 1, "is_match" : int(prediction[i]), "confidence" : float(probability[i])} for i in range(len(multi_input_data))]}
        
        except Exception as err:
            return {"status": 0, "message" : str(err)}
    