from typing import Tuple
import Utilities as utils
import os
from classifiers.LaplacianMRClassifier import LaplacianMRClassifier
from contracts.v1.post_search import PostSearch

class PostSearchImpl(PostSearch):
    def do_post_search_operations(self, best_hyperparam_point: dict,
                                  preprocessing_output: dict,
                                  model_path: str) -> Tuple[dict, str]:
        
        model_path = os.path.join(model_path, 'model.pkl')
        
        model = LaplacianMRClassifier(**best_hyperparam_point)
        model.fit(preprocessing_output['x_train_l'], preprocessing_output['labels_train'], preprocessing_output['x_train_u'])
        
        metrics = model.metrics(preprocessing_output['x_test'], preprocessing_output['labels_test'], preprocessing_output['class_names'])
        score = model.score(preprocessing_output['x_test'], preprocessing_output['labels_test'])
        
        utils.save_data_pkl(model, model_path)
        msg = f"model saved as : {model_path}"

        return {'results': score, 'model': model_path, 'metrics': metrics}, msg
