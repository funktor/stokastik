from typing import Tuple
from contracts.v1.scoring import Scoring
from scipy.stats import uniform, randint
from classifiers.LaplacianMRClassifier import LaplacianMRClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.gaussian_process.kernels import RBF

class ScoringImpl(Scoring):
    def score_hyperparameter_point(self, sampled_hyperparam_point:dict, preprocessing_output:dict) -> Tuple[float, str]:
        model = LaplacianMRClassifier(**sampled_hyperparam_point)
        model.fit(preprocessing_output['x_train_l'], preprocessing_output['labels_train'], preprocessing_output['x_train_u'])
        score = model.score(preprocessing_output['x_train_l'], preprocessing_output['labels_train'])

        return score, "Hello world"