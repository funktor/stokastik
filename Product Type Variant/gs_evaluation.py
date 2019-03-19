import siamese_api
from siamese_api import SiameseAPI
import data_generator as dg
import pandas as pd, random, pickle, numpy as np, tables, sys, time, collections, math, os
from sklearn.neighbors import KDTree
import grouping_utils as gutils
import constants as cnt

try:
    validation_set = pd.read_csv("data/grouping_validation_set.csv")
    gs_items = pd.read_csv("data/items_gs.csv")

    items = gutils.load_data_pkl('items.pkl')
    item_id_index_map = dict()
    for i in range(len(items)):
        item_id_index_map[items[i][0]] = i

    base_item_ids = list(validation_set['Item Id'])
    grouped_item_ids = list(validation_set['Items to be grouped'])

    gs_mapping = collections.defaultdict(list)
    for i in range(len(base_item_ids)):
        if grouped_item_ids[i] in item_id_index_map:
            gs_mapping[base_item_ids[i]].append(grouped_item_ids[i])

    sapi = SiameseAPI()
    sapi.get_model()
    sapi.model.init_model()
    sapi.model.load()

    kdtree = gutils.load_data_pkl('kd_tree.pkl')

    embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, cnt.SIAMESE_EMBEDDINGS_FILE), mode='r')
    embeds_arr = embeds_file.root.data
    
    for thres in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
        radius = sapi.get_distance_threshold(threshold=thres)

        num_predicted, num_actual, num_correct = 0, 0, 0

        for base_item_id, grouped_item_ids in gs_mapping.items():
            if base_item_id in item_id_index_map:
                rep = embeds_arr[item_id_index_map[base_item_id]]

                nearest = gutils.get_nearest_neighbors_radius(kdtree, rep, query_radius=radius)
                nearest = [int(x) for x in nearest]
                predicted_item_ids = set([items[x][0] for x in nearest if math.isnan(items[x][0]) is False])

                grouped_item_ids = set(grouped_item_ids)
                a, b, c = len(grouped_item_ids.intersection(predicted_item_ids)), len(predicted_item_ids), len(grouped_item_ids)

                num_correct += a
                num_predicted += b
                num_actual += c

#                 print(a, b, c)

#                 print("Base item id : " , base_item_id)
#                 print("Base item title : " , gs_item_title_map[base_item_id])
#                 print("Total actual : " , c)
#                 print("Total predicted : " , b)
#                 print()
#                 print("All actual items")
#                 for x in grouped_item_ids:
#                     if x in gs_item_title_map:
#                         print(gs_item_title_map[x])

#                 print()
#                 print("All predicted items")
#                 for x in predicted_item_ids:
#                     print(item_id_title_map[x])
#                 print()
#                 print()

        precision = float(num_correct)/num_predicted
        recall = float(num_correct)/num_actual

        print(thres, radius, precision, recall)
finally:
    embeds_file.close()
