from typing import Tuple

class PostSearch:

    def do_post_search_operations(self, best_hyperparam_point:dict, preprocessing_output:dict, model_folder:str)->Tuple[dict, str]:
        raise NotImplementedError
