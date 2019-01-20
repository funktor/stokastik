import sys
import pandas as pd
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import re, time
from scipy.sparse import hstack
import os

if len(sys.argv) > 1 and str(sys.argv[1]) == 'cython':
    from CustomFeatureTransformer import AreaRugTransformer
    from CustomClassifier import AreaRugClassifier
else:
    from CustomFeatureTransformerPy import AreaRugTransformer
    from CustomClassifierPy import AreaRugClassifier

np.random.seed(0)
df = pd.read_csv("/Users/a0m02fp/area_rug_tagged_data.csv")

df[u'product_name'].apply(str)
df[u'product_short_description'].apply(str)
df[u'product_long_description'].apply(str)
df[u'Manual curation'].apply(str)

df_title_data = lambda dataF: list(dataF[u'product_name'])
df_desc_data = lambda dataF: list(dataF[u'product_short_description'] + " " + dataF[u'product_long_description'])

msk = np.random.rand(len(df)) < 0.8

train_df, test_df = df[msk], df[~msk]

train_sents_title, test_sents_title = map(lambda x:str(x), df_title_data(train_df)), map(lambda x:str(x), df_title_data(test_df))
train_sents_desc, test_sents_desc = map(lambda x:str(x), df_desc_data(train_df)), map(lambda x:str(x), df_desc_data(test_df))

train_labels, test_labels = list(train_df[u'Manual curation']), list(test_df[u'Manual curation'])

start = time.time()
transformer_title, transformer_desc = AreaRugTransformer(num_features=30, max_ngram=2), AreaRugTransformer(num_features=30, max_ngram=2)

transformer_title.fit(train_sents_title, train_labels)
transformer_desc.fit(train_sents_desc, train_labels)

train_title_data, train_desc_data = transformer_title.transform(train_sents_title), transformer_desc.transform(train_sents_desc)

train_data = hstack((train_title_data, train_desc_data))
print time.time()-start

start = time.time()
base_estimator = LogisticRegression(penalty='l1')

model = AreaRugClassifier(base_estimator, train_title_data.shape[1], predict_threshold=0.5)
model.fit(train_data, train_labels)

test_title_data, test_desc_data = transformer_title.transform(test_sents_title), transformer_desc.transform(test_sents_desc)

test_data = hstack((test_title_data, test_desc_data))

print model.score(test_data, test_labels)
print time.time()-start

start = time.time()
test_title_data, test_desc_data = transformer_title.transform(test_sents_title[0:1]), transformer_desc.transform(test_sents_desc[0:1])
test_data = hstack((test_title_data, test_desc_data))
model.predict(test_data)
print time.time()-start