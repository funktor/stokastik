from typing import Tuple


class Scoring:

    def score_hyperparameter_point(self, sampled_hyperparam_point:dict, preprocessing_output:dict)->Tuple[float,str]:
        raise NotImplementedError

    def multi_metric_score_hyperparameter_point(self, sampled_hyperparam_point:dict, preprocessing_output:dict)->Tuple[dict,str]:
        raise NotImplementedError
