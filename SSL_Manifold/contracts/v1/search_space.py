from typing import Tuple

class SearchSpace:

    def construct_and_return_search_space(self, preprocessing_output: dict) -> Tuple[dict, str]:
        raise NotImplementedError
