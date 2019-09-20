import os, numpy as np, pandas as pd
import utilities.color_extraction.utilities as utils
import shared_utilities as shutils
from networks.color_extraction.network import ColorExtractionNetwork
import constants.color_extraction.constants as cnt

def get_input_data_from_image(image):
    image = shutils.process_image(image, cnt.IMAGE_SIZE)
    return np.array([shutils.image_to_array(image)])

class Inference:
    def __init__(self):
        self.network = ColorExtractionNetwork()
        self.network.init_model()
        self.network.load()
        self.pt_encoder = shutils.load_data_pkl(cnt.PT_ENCODER_PATH)
        self.cl_encoder = shutils.load_data_pkl(cnt.COLOR_ENCODER_PATH)

    def predict(self, image):
        try:
            input_data = get_input_data_from_image(image)
            
            cl_prediction, cl_probability = self.network.predict(input_data, type='color', return_probability=True)
            pt_prediction, pt_probability = self.network.predict(input_data, type='pt', return_probability=True)
            
            cl_prediction = self.cl_encoder.inverse_transform(np.array(cl_prediction))
            
            if np.sum(pt_prediction) == 0:
                pt_prediction = [[]]
            else:
                pt_prediction = self.pt_encoder.inverse_transform(np.array(pt_prediction))

            return {"status": 1, "product_type" : pt_prediction[0], "product_type_confidence" : pt_probability[0], "color" : list(cl_prediction[0]), "color_confidence" : cl_probability[0]}
        
        except Exception as err:
            return {"status": 0, "message" : str(err)}