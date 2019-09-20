from typing import Tuple
from contracts.v1.search_space import SearchSpace
from scipy.stats import uniform, randint
from sklearn.gaussian_process.kernels import RBF
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import numpy as np

class SearchSpaceImpl(SearchSpace):
    def construct_and_return_search_space(self, preprocessing_output: dict) -> Tuple[dict, str]:
#         grid = {'algorithms': ['RLS'], 'n_neighbors': 7, 'kernel': RBF(1.0), 'lambda_k': 0.000005, 'lambda_u': 0.001, 'u_split': 0.0, 'threshold': 0.0, 'strategy': 'IC'}
        
        grid = {'n_neighbors': hp.choice('n_neighbors', list(range(1, 11))),
                'lambda_k': hp.choice('lambda_k', (np.arange(1,10**4)*10**-6).tolist()), 
                'lambda_u': hp.choice('lambda_u', (np.arange(1,10**3)*10**-4).tolist()), 
                'kernel': hp.choice('kernel', [RBF(1,(1,10))]), 
                'u_split': hp.choice('u_split', [0.0]), 
                'threshold': hp.choice('threshold', [0.0]), 
                'algorithms': hp.choice('algorithms', [['RLS'], ['RLR'], ['LapSVM']]), 
                'strategy': hp.choice('strategy', ['IC', 'OVA'])
               }

        return grid, f"Search space constructed ..."
