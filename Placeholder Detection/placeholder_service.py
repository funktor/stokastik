from hashing_service import HashingService
from siamese_service import SiameseService
import json, logging, collections, time
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
        
    def predict(self, request):
        try:
            self.load()
            
            url_identifiers = dict()
            for req in request:
                for url in req['secondaryURL']:
                    url_identifiers[url] = {'url': url, 
                                            'product_type':req['product_type'], 
                                            'productId':req['product_id'], 
                                            'itemId':req['item_id'], 
                                            'imageType':'secondaryURL', 
                                            'tenantId': '0', 
                                            'classification_tag': '',
                                            'classifierType': '',
                                            'classifierVersion': '2.0', 
                                            'imageDownloadTimeInSecs': 0,
                                            'processingTimeInSecs': 0,
                                            'source': 'Placeholder', 
                                            'tags':{}}
                    
                for url in req['primaryURL']:
                    url_identifiers[url] = {'url': url, 
                                            'product_type':req['product_type'], 
                                            'productId':req['product_id'], 
                                            'itemId':req['item_id'], 
                                            'imageType':'primaryURL', 
                                            'tenantId': '0', 
                                            'classification_tag': '',
                                            'classifierType': '',
                                            'classifierVersion': '2.0', 
                                            'imageDownloadTimeInSecs': 0,
                                            'processingTimeInSecs': 0,
                                            'source': 'Placeholder', 
                                            'tags':{}}

            urls = list(url_identifiers.keys())
            
            imgs, valid_urls = [], []
            for url in urls:
                start = time.time()
                img = utils.image_url_to_obj(url)
                duration = time.time()-start
                
                if isinstance(img, int) is False:
                    imgs.append(img)
                    url_identifiers[url]['imageDownloadTimeInSecs'] = duration
                    valid_urls.append(url)
            
            prediction = self.hash_service.predict(valid_urls, imgs, url_identifiers)
            
            if prediction == 0:
                return json.dumps({'status':'failure', 'message':'Some error occurred'})
            
            else:
                img_arrs, valid_urls2 = [], []
                
                for i in range(len(imgs)):
                    if len(url_identifiers[valid_urls[i]]['tags']) == 0:
                        start = time.time()
                        img_arr = utils.image_to_array(imgs[i])
                        duration = time.time()-start
                        
                        if isinstance(img_arr, int) is False:
                            img_arrs.append(img_arr)
                            url_identifiers[valid_urls[i]]['processingTimeInSecs'] += duration
                            valid_urls2.append(valid_urls[i])
                
                prediction = self.siam_service.predict(valid_urls2, img_arrs, url_identifiers)
                
                if prediction == 0:
                    return json.dumps({'status':'failure', 'message':'Some error occurred'})
            
            return [url_identifiers[url] for url in urls]
            
        except Exception as err:
            return json.dumps({'status':'failure', 'message':err.message})