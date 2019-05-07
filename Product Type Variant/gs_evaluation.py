import siamese_api
from siamese_api import SiameseAPI
import data_generator as dg
import pandas as pd, random, pickle, numpy as np, tables, sys, time, collections, math, os
from sklearn.neighbors import KDTree
import grouping_utils as gutils
import constants as cnt
from sklearn.metrics.pairwise import euclidean_distances

try:
    validation_set = pd.read_csv("data/grouping_validation_set.csv")

    items = gutils.load_data_pkl('items_13.pkl')
    item_id_index_map = dict()
    for i in range(len(items)):
        item_id_index_map[items[i][0]] = i
        
    pt_indices = collections.defaultdict(list)
    for i in range(len(items)):
        pt = items[i][1]
        pt_indices[pt].append(i)

    base_item_ids = list(validation_set['Item Id'])
    grouped_item_ids = list(validation_set['Items to be grouped'])
    
    print(len(set(base_item_ids)))
    print(len(set(grouped_item_ids)))
    
    print(len(set(base_item_ids).intersection(set([x for x, y in item_id_index_map.items()]))))
    print(len(set(grouped_item_ids).intersection(set([x for x, y in item_id_index_map.items()]))))

    gs_mapping = collections.defaultdict(list)
    for i in range(len(base_item_ids)):
        if grouped_item_ids[i] in item_id_index_map:
            gs_mapping[base_item_ids[i]].append(grouped_item_ids[i])

    sapi = SiameseAPI()
    sapi.get_model()
    sapi.model.init_model()
    sapi.model.load()
    
    if cnt.INDEXING_DS == 'KDTREE':
        ds = gutils.load_data_pkl('kd_tree_new_13_pt.pkl')
    else:
        ds = gutils.load_data_pkl('pq.pkl')

    embeds_file = tables.open_file(os.path.join(cnt.DATA_FOLDER, 'embeddings_13.h5'), mode='r')
    embeds_arr = embeds_file.root.data
    
    correct_pt_threshold, correct_pt_threshold = collections.defaultdict(dict), collections.defaultdict(dict)
    pred_pt_threshold, pred_pt_threshold = collections.defaultdict(dict), collections.defaultdict(dict)
    actual_pt_threshold, actual_pt_threshold = collections.defaultdict(dict), collections.defaultdict(dict)
    
    for thres in [0.9995]:
        radius = sapi.get_distance_threshold(threshold=thres)

        num_predicted, num_actual, num_correct = 0, 0, 0

        for base_item_id, grouped_item_ids in gs_mapping.items():
            if base_item_id in item_id_index_map:
                pt = items[item_id_index_map[base_item_id]][1]
                pt_idx = pt_indices[pt]
                
                if thres not in correct_pt_threshold[pt]:
                    correct_pt_threshold[pt][thres] = 0

                if thres not in pred_pt_threshold[pt]:
                    pred_pt_threshold[pt][thres] = 0

                if thres not in actual_pt_threshold[pt]:
                    actual_pt_threshold[pt][thres] = 0
                
                rep = embeds_arr[item_id_index_map[base_item_id]]
                grouped_item_ids = set(grouped_item_ids)

                if cnt.INDEXING_DS == 'KDTREE':
                    nearest = gutils.get_nearest_neighbors_radius(ds[pt], rep, query_radius=radius)
                else:
                    nearest = gutils.get_nearest_neighbors_radius_pq(ds[pt], rep, query_radius=0.4)
                    
                if len(nearest) > 0:
                    nearest = [pt_idx[int(x)] for x in nearest]

                    predicted_item_ids = set([items[x][0] for x in nearest if math.isnan(items[x][0]) is False])
                    print(len(predicted_item_ids), len(grouped_item_ids))
                    
                    a, b, c = len(grouped_item_ids.intersection(predicted_item_ids)), len(predicted_item_ids), len(grouped_item_ids)
                    
                    correct_pt_threshold[pt][thres] += a
                    pred_pt_threshold[pt][thres] += b
                    actual_pt_threshold[pt][thres] += c
                    
                    num_correct += a
                    num_predicted += b
                    num_actual += c
                    
                    file_name = os.path.join('gs_preds', 'pred_out_' + str(int(base_item_id)) + '.txt')
                    
                    with open(file_name, 'w') as f:
                        j = item_id_index_map[base_item_id]
                        
                        f.write("Query Item Id : " + str(int(base_item_id)) + "\n")
                        f.write("Query Text : " + str(items[j][2]) + " " + str(items[j][4]) + "\n")
                        
                        f.write("\n\n")
                        
                        f.write("Predicted:\n")
                        for x in predicted_item_ids:
                            j = item_id_index_map[x]
                            f.write("ItemId : " + str(x) + "\n")
                            f.write("Title-Desc : " + str(items[j][2]) + " " + str(items[j][4]) + "\n")
                            f.write("\n")
                        f.write("\n")
                        
                        f.write("Actual:\n")
                        for x in grouped_item_ids:
                            j = item_id_index_map[x]
                            f.write("ItemId : " + str(x) + "\n")
                            f.write("Title-Desc : " + str(items[j][2]) + " " + str(items[j][4]) + "\n")
                            f.write("\n")
                            
                        f.write("\n\n")
                        
                else:
                    a, b, c = 0, 0, len(grouped_item_ids)
                    
                    correct_pt_threshold[pt][thres] += a
                    pred_pt_threshold[pt][thres] += b
                    actual_pt_threshold[pt][thres] += c

                    num_correct += a
                    num_predicted += b
                    num_actual += c

        if num_predicted > 0:
            precision = float(num_correct)/num_predicted
            recall = float(num_correct)/num_actual

            print(thres, radius, precision, recall)
            
    gutils.save_data_pkl(correct_pt_threshold, 'correct_pt_threshold.pkl')
    gutils.save_data_pkl(pred_pt_threshold, 'pred_pt_threshold.pkl')
    gutils.save_data_pkl(actual_pt_threshold, 'actual_pt_threshold.pkl')
    
    for pt in actual_pt_threshold:
        print(pt)
        for thres in actual_pt_threshold[pt]:
            if pt in correct_pt_threshold and thres in correct_pt_threshold[pt]:
                if pt in pred_pt_threshold and thres in pred_pt_threshold[pt] and pred_pt_threshold[pt][thres] > 0:
                    precision = float(correct_pt_threshold[pt][thres])/pred_pt_threshold[pt][thres]
                else:
                    precision = 0.0

                if pt in actual_pt_threshold and thres in actual_pt_threshold[pt] and actual_pt_threshold[pt][thres] > 0:
                    recall = float(correct_pt_threshold[pt][thres])/actual_pt_threshold[pt][thres]
                else:
                    recall = 0.0
            else:
                precision, recall = 0.0, 0.0
                
            print(thres, precision, recall)
        print()
    print()
    
finally:
    embeds_file.close()