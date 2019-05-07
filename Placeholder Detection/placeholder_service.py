from hashing_service import HashingService
from siamese_service import SiameseService
import json, logging
import utilities as utils

class PlaceholderService(object):
    def __init__(self):
        self.hash_service = HashingService()
        self.siam_service = SiameseService()
        
    def train(self):
        self.hash_service.train()
        self.siam_service.train()
    
    def evaluate(self):
        self.hash_service.evaluate()
        self.siam_service.evaluate()
    
    def save(self):
        self.hash_service.save()
        self.siam_service.save()
    
    def load(self):
        self.hash_service.load()
        self.siam_service.load()
        
    def predict(self, urls):
        try:
            self.load()

            imgs = [utils.image_url_to_obj(url) for url in urls]
            
            prediction = json.loads(self.hash_service.predict(urls, imgs))
            
            if prediction['status'] == 'failure':
                return json.dumps({'status':'failure', 'message':'Some error occurred'})
            
            else:
                output = prediction['output']
                pl_output = [out for out in output if out['label'] == 1]
                
                non_pl_indices = [out['index'] for out in output if out['label'] == 0]
                
                img_arrs = [utils.image_to_array(imgs[i]) for i in non_pl_indices]
                non_pl_urls = [urls[i] for i in non_pl_indices]
                
                prediction = json.loads(self.siam_service.predict(non_pl_urls, img_arrs))
                
                dl_output = []
                
                if prediction['status'] != 'failure':
                    dl_output = prediction['output']
                
                return json.dumps({'output':pl_output + dl_output, 'status':'completed'})
            
        except Exception as err:
            return json.dumps({'status':'failure', 'message':err.message})