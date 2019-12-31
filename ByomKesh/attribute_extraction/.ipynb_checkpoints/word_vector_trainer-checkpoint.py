import sys
sys.path.append('/home/jupyter/MySuperMarket/')
from feature_transformers.W2V_Transformer import W2VFeatures 
import pandas as pd
import Utilities as utils

chunksize = 10 ** 6
i = 0
for df_chunk in pd.read_csv('data/og_pcfs_pts.tsv', sep='\\t', engine='python', error_bad_lines=False, chunksize=chunksize):
    print(i)
    title = df_chunk.title.str.lower().apply(str)
    short_desc = df_chunk.short_description.str.lower().apply(str)
    long_desc = df_chunk.long_description.str.lower().apply(str)
    
    sentences = list(title + " " + short_desc + " " + long_desc)
    
    wv = W2VFeatures(num_components=128, features=None, wv_type='W2V')
    wv.fit(sentences)
    utils.save_data_pkl(wv, 'persistence/wv_models/word_vector_' + str(i) + ".pkl")
    
    i += 1