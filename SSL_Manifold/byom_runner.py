import os, sys
curr_wd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_wd)

from byom_scripts.preprocessing_impl import PreprocessingImpl
from byom_scripts.search_space_impl import SearchSpaceImpl
from byom_scripts.scoring_impl import ScoringImpl
from byom_scripts.post_search_impl import PostSearchImpl
from sklearn.gaussian_process.kernels import RBF
from Predictor import Predictor

user_inputs = {'attribute': 'age_group','num_l': 4000,'num_u': 4000,'num_features_title': 500,'num_features_desc': 500,'use_pca': True,'embed_dim_title': 128,'embed_dim_desc': 128,'use_unlabelled_features': True,'use_unlabelled_training': True}

preprocess_obj = PreprocessingImpl()
preprocessed_data, msg = preprocess_obj.do_preprocessing_and_return_preprocessed_variables(user_inputs)

sampled_hyperparam_point = {'algorithms': ['RLS'], 'n_neighbors': 4, 'kernel': RBF(1.0), 'lambda_k': 0.00005, 'lambda_u': 0.2, 'u_split': 0.0, 'threshold': 0.5, 'strategy': 'IC'}

# scoring_obj = ScoringImpl()
# score, msg = scoring_obj.score_hyperparameter_point(sampled_hyperparam_point, preprocessed_data)

post_search_obj = PostSearchImpl()
output, msg = post_search_obj.do_post_search_operations(sampled_hyperparam_point, preprocessed_data, "/data/ssl_manifold")

print(output)

item_data = {'Title': '10k Black Hills Gold Men&#39;s Claddagh Ring size 14.5', 
             'Short Description': 'Attributes:<br>Casted<br>Diamond-cut<br>Polished<br>Satin<br>10K Yellow gold<br>12k leaf accents<br>Mens<br>', 
             'Long Description': 'Product Type: Jewelry<br>Jewelry Type: Rings<br>Material: Primary: Gold<br>Material: Primary - Color: Yellow<br>Material: Primary - Purity: 10K<br>Width: 2 mm<br>Sold By Unit: Each<br>Ring Top Length: 12<br>Ring Top Length U/M: mm<br><br>'}

p = Predictor("/data/ssl_manifold/model.pkl", user_inputs)
print(p.predict(item_data))
