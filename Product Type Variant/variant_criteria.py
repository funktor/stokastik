import common_utils as utils
from collections import defaultdict
import numpy as np, random, re, math
from itertools import combinations

class Variant(object):
    def __init__(self, clusters, items, min_cluster_size=5, max_attrs_per_var=5, max_variants=100, 
                 valid_attrs=None, valid_variants=None, excluded_attrs=None):
        self.clusters = clusters
        self.items = items
        self.max_attrs_per_var = max_attrs_per_var
        self.max_variants = max_variants
        self.min_cluster_size = min_cluster_size
        self.valid_attrs = valid_attrs
        self.valid_variants = valid_variants
        self.excluded_attrs = excluded_attrs
        
    def get_attribute_scores(self, item_indexes):
        attribute_value_cnts = defaultdict(set)
        n = len(item_indexes)

        for x in item_indexes:
            attributes = self.items[x][5]

            if self.valid_attrs is not None and len(self.valid_attrs) > 0:
                keys = sorted([key for key in attributes if key in self.valid_attrs.difference(self.excluded_attrs)])
                
            elif self.excluded_attrs is not None and len(self.excluded_attrs) > 0:
                keys = sorted([key for key in attributes if key not in self.excluded_attrs])
                
            else:
                keys = sorted([key for key in attributes])

            vals = [attributes[key] for key in keys]

            k_combinations, v_combinations = [], []

            for i in range(1, self.max_attrs_per_var + 1):
                k_combinations += list(combinations(keys, i))
                v_combinations += list(combinations(vals, i))

            if self.valid_variants is not None and len(self.valid_variants) > 0:
                valid_combinations = [(key, val) for key, val in zip(k_combinations, v_combinations) if key in self.valid_variants]
            else:
                valid_combinations = zip(k_combinations, v_combinations)

            for key, val in valid_combinations:
                attribute_value_cnts[key].add(val)

        attribute_scores = defaultdict(float)

        for attr, values in attribute_value_cnts.items():
            attribute_scores[attr] = float(len(values))/n

        return attribute_scores
    
    def get_variant_scores(self):
        overall_scores = defaultdict(float)
        total = np.sum([len(indexes) for label, indexes in self.clusters.items() if len(indexes) >= self.min_cluster_size])

        for label, indexes in self.clusters.items():
            if len(indexes) >= self.min_cluster_size:
                n, scores = len(indexes), self.get_attribute_scores(indexes)

                for attr, score in scores.items():
                    overall_scores[attr] += n * score/float(total)

        out = sorted([(attr, score) for attr, score in overall_scores.items()], key=lambda k:-k[1])
        min_score = out[min(len(out), self.max_variants)-1][1]
        
        return [(x, y) for x, y in out if y >= min_score]
    
    def get_predicted_variants(self):
        output = defaultdict()

        for label, indexes in self.clusters.items():
            if len(indexes) >= self.min_cluster_size:
                highest = []
                scores = self.get_attribute_scores(indexes)

                if len(scores) > 0:
                    max_score = max([y for x, y in scores.items()])

                    for x, y in scores.items():
                        if y >= max_score:
                            if isinstance(x, tuple):
                                x = tuple(sorted(x))
                            highest.append(x)

                output[label] = highest

        return output
    
    def results(self, predicted_variants, predominant_variant):
        predominant_variant = predominant_variant[0] if len(predominant_variant) == 1 else predominant_variant
        
        tp, tn, fp, fn = 0, 0, 0, 0
        accuracy = 0

        for label, indexes in self.clusters.items():
            if len(indexes) >= self.min_cluster_size:
                pred = predicted_variants[label]
                pred = map(lambda x:x[0] if len(x) == 1 else x, pred)

                true_var_cnt = defaultdict(float)

                for idx in indexes:
                    key = [x for x, y in self.items[idx][7].items()]
                    if len(key) > 0:
                        if len(key) > 1:
                            key = tuple(sorted(key))
                        else:
                            key = key[0]
                            
                        if key in pred:
                            accuracy += 1
                            
                        if predominant_variant in pred and key == predominant_variant:
                            tp += 1
                        elif predominant_variant in pred and key != predominant_variant:
                            fp += 1
                        elif predominant_variant not in pred and key == predominant_variant:
                            fn += 1
                        else:
                            tn += 1
                            
        accuracy /= float(tp + fp + tn + fn)
        precision = float(tp)/(tp + fp) if tp + fp > 0 else 0
        recall = float(tp)/(tp + fn) if tp + fn > 0 else 0

        f1_score = float(2*precision*recall)/(precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1_score, accuracy

