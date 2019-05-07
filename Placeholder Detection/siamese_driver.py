import constants as cnt
import utilities as utils
from siamese_predictor import SiameseModel

def test_model():
    model = utils.load_siamese_model()
    model.score()

def evaluate():
    model = SiameseModel()
    model.fit()
    model.save()
    model.score()
    
evaluate()
test_model()

utils.compute_embeddings()
utils.ensemble_score()