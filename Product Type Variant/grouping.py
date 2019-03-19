from collections import defaultdict
import numpy as np, random, re, math
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import data_generator as dg

class Grouping(object):
    def __init__(self, items, representations=None, similarity_threshold=1.0):
        self.items = items
        self.true_groups = None
        self.auto_groups = None
        self.representations = representations
        self.confidence = similarity_threshold
        
    def true_grouping(self):
        clusters = defaultdict(list)

        for idx in range(len(self.items)):
            abs_pd_id = self.items[idx][5]
            clusters[abs_pd_id].append(idx)

        return clusters
    
    def cluster_on_brands(self):
        brands = defaultdict(list)

        for idx in range(len(self.items)):
            attributes = self.items[idx][5]
            if 'brand' in attributes:
                brands[attributes['brand']].append(idx)
            else:
                brands['unknown'].append(idx)

        return brands
    
    def cluster_on_pts_abs_id(self):
        pts = defaultdict(set)

        for idx in range(len(self.items)):
            pt, abs_id = self.items[idx][1], self.items[idx][5]
            pts[pt].add(abs_id)

        return pts
    
    def kmeans(self):
        clusters = defaultdict(list)
        label_start = 0

        for pt, abs_ids in pre_clusters.items():
            indexes = [groups[x][0] for abs_id in abs_ids]
            for abs_id in abs_ids:
                indexes += y
                
            data = self.representations[indexes,:]

            num_clusters = int(0.5*len(abs_id_indexes))
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=min(1000, data.shape[0]), init='k-means++', init_size=num_clusters+1)
            kmeans.fit(data)

            for idx in range(len(kmeans.labels_)):
                label = label_start + kmeans.labels_[idx]
                clusters[label].append(indexes[idx])

            label_start += max(kmeans.labels_) + 1

        return clusters
    
    def auto_grouping(self):
        try:
            groups = self.true_grouping()
        
            pre_clusters = self.cluster_on_pts_abs_id()
            pt_cluster_ids, pt_clusters = defaultdict(dict), defaultdict(dict)

            for pt, abs_ids in pre_clusters.items():
                print(pt)
                print(len(abs_ids))
                
                abs_ids = list(abs_ids)
                item_text = []
                for abs_id in abs_ids:
                    x = groups[abs_id][0]
                    item_text.append(dg.get_item_text(self.items[x]))

                embeds_file = tables.open_file('data/w2v_embeddings.h5', mode='r')
                data = embeds_file.root.data

                num_clusters = int(0.5*len(abs_ids))
                kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=min(1000, data.shape[0]), init='k-means++', init_size=num_clusters+1)
                kmeans.fit(data)

                for i, label in enumerate(kmeans.labels_):
                    pt_cluster_ids[pt][abs_ids[i]] = label

                    if label not in pt_clusters[pt]:
                        pt_clusters[pt][label] = []

                    pt_clusters[pt][label].append(abs_ids[i])

            dg.save_data_pkl(pt_cluster_ids, 'pt_cluster_ids.pkl')
            dg.save_data_pkl(pt_clusters, 'pt_clusters.pkl')

            return pt_cluster_ids, pt_clusters
                
        finally:
            embeds_file.close()
            
        
        
    def init_groups(self):
        print("Getting actual groups...")
        self.true_groups = self.true_grouping()
        
        print("Getting auto groups...")
        self.auto_groups = self.auto_grouping()

    def get_cluster_score(self, cluster):
        counts = defaultdict(float)
        n = len(cluster)

        for x in cluster:
            counts[self.items[x][6]] += 1.0/n

        ent = [-p*np.log2(p) for k, p in counts.items()]

        return np.sum(ent)

    def get_clustering_scores(self):
        scores = [len(indices) * self.get_cluster_score(indices) for cluster_id, indices in self.auto_groups.items()]
        return np.sum(scores)/float(len(self.items))