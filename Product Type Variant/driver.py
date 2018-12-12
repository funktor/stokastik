import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

import pandas as pd
import common_utils as utils
from grouping import Grouping
from variant_criteria import Variant
import json
from siamese_api import SiameseAPI
import numpy as np, time, random

def get_variants(items, groups, excluded_attrs):
    print "Fetching important attributes..."
    var_instance = Variant(groups, items, max_attrs_per_var=1, max_variants=10, excluded_attrs=excluded_attrs)
    valid_attrs = set([attr[0] for attr, score in var_instance.get_variant_scores()])

    print "Fetching variant criteria..."
    var_instance = Variant(groups, items, max_attrs_per_var=2, max_variants=1, excluded_attrs=excluded_attrs, valid_attrs=valid_attrs)
    variant_scores = var_instance.get_variant_scores()

    print "Predominant Variant Criteria : ", variant_scores[0]

    print "Fetching variant criteria per group..."
    pred_grp_variants = var_instance.get_predicted_variants()

    print "Results"
    print var_instance.results(pred_grp_variants, variant_scores[0][0])
    

file_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/a0m02fp/stormbreaker/product_type_variant/items_data.csv'
item_type = sys.argv[2] if len(sys.argv) > 2 else 'Bras'
group_type = sys.argv[3] if len(sys.argv) > 3 else 'auto'
max_num_attrs = int(sys.argv[4]) if len(sys.argv) > 4 else 2

print "Reading data file..."

df = pd.read_csv(file_path, sep=",", encoding='utf-8')
df['attr_val_pairs'] = df['attr_val_pairs'].apply(lambda x: json.loads(x))
df['variant_criterias'] = df['variant_criterias'].apply(lambda x: json.loads(x))

print "Reading product data..."
product_data = list(df.itertuples(index=False))

print "Getting items..."
items = utils.get_unique_items_pt(product_data, item_type)

print "Creating groups..."
grp_instance = Grouping(items)
grp_instance.init_groups()

print "Cluster entropy score = ", grp_instance.get_clustering_scores()

groups1 = grp_instance.auto_groups if group_type == 'auto' else grp_instance.true_groups
    
print "Reading excluded attribute list..."
with open('excluded_attr_list.txt', 'rb') as x_attr:
    excluded_attrs = x_attr.readlines()

excluded_attrs = [x.strip() for x in excluded_attrs] 
excluded_attrs = set(excluded_attrs)

print "Getting variants..."
get_variants(items, groups1, excluded_attrs)
print

print "Training siamese network model..."
sapi = SiameseAPI(items, grp_instance.true_groups, max_num_tokens=50)
sapi.train_model()

print "Getting siamese network representations..."
embeds = sapi.get_representations([str(x[2]) for x in items])

print "Clustering using siamese representations..."
grp_instance = Grouping(items, representations=embeds)
grp_instance.init_groups()
groups2 = grp_instance.auto_groups

print "Siamese network cluster entropy score = ", grp_instance.get_clustering_scores()

print "Getting variants..."
get_variants(items, groups2, excluded_attrs)
print