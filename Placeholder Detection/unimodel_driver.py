import importlib
import data_generator as dg
importlib.reload(dg)
import unimodel_classifier
importlib.reload(unimodel_classifier)
from unimodel_classifier import UniModel

use_vgg = True
best_model = True

model = UniModel('data/unimodel.h5', 
                 'data/unimodel_current_best.h5', 
                 batch_size=256, 
                 training_samples=100000, 
                 validation_samples=10000, 
                 testing_samples=10000, 
                 use_vgg=use_vgg)

def test_model():
    model.init_model()
    model.load(best_model=best_model)
    model.score()

def evaluate():
    model.fit()
    model.save()
    model.score()
    
evaluate()