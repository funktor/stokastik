import Utilities as utils
import Constants as cnt
import os, numpy as np

class Predictor(object):
    def __init__(self, model_path, user_inputs):
        self.model = utils.load_data_pkl(model_path)
        
        attribute = user_inputs['attribute'] if 'attribute' in user_inputs else 'fit'
        num_l = user_inputs['num_l'] if 'num_l' in user_inputs else 1000
        num_u = user_inputs['num_u'] if 'num_u' in user_inputs else 500
        
        feature_title_path = os.path.join(cnt.PERSISTENCE_PATH, attribute + '_' + str(num_l) + '_TITLE.pkl')
        feature_desc_path = os.path.join(cnt.PERSISTENCE_PATH, attribute + '_' + str(num_l) + '_DESC.pkl')
        label_transformer_path = os.path.join(cnt.PERSISTENCE_PATH, attribute + '_' + str(num_l) + '_LABEL_TRANSFORMER.pkl')
        
        self.feature_tf_title = utils.load_data_pkl(feature_title_path)
        self.feature_tf_desc = utils.load_data_pkl(feature_desc_path)
        self.label_transformer = utils.load_data_pkl(label_transformer_path)
        
        
    def predict_proba(self, item_data):
        title = item_data['Title']
        desc = item_data['Short Description'] + " " + item_data['Long Description']
        
        titles_test = np.array([title])
        descriptions_test = np.array([desc])
        
        x_test_title = self.feature_tf_title.transform(titles_test)
        x_test_desc = self.feature_tf_desc.transform(descriptions_test)

        x_test = np.hstack((x_test_title, x_test_desc))
        preds = self.model.predict(x_test)
        
        return preds
        
        
    def predict(self, item_data):
        preds = self.predict_proba(item_data)
        predicted_class = self.label_transformer.inverse_transform(preds)[0]
        
        return '__'.join(predicted_class)
    
    
    def predict_batch(self, item_data):
        title = list(item_data['Title'])
        desc = list(item_data['Short Description'] + " " + item_data['Long Description'])
        
        titles_test = np.array(title)
        descriptions_test = np.array(desc)
        
        x_test_title = self.feature_tf_title.transform(titles_test)
        x_test_desc = self.feature_tf_desc.transform(descriptions_test)

        x_test = np.hstack((x_test_title, x_test_desc))
        preds = self.model.predict(x_test)
        
        predicted_classes = self.label_transformer.inverse_transform(preds)
        
        return ['__'.join(x) for x in predicted_classes]