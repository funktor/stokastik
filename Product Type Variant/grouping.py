import common_utils as utils
from collections import defaultdict
import numpy as np, random, re, math
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from embeddings import DL_API
from sklearn.metrics.pairwise import cosine_similarity

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
            abs_pd_id = self.items[idx][6]
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
    
    def kmeans(self, brand_clusters):
        clusters = defaultdict(list)
        label_start = 0

        for brand, indexes in brand_clusters.items():
            data = self.representations[indexes,:]

            num_clusters = int(math.ceil(data.shape[0]/2.0))
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=min(1000, data.shape[0]), init='k-means++')
            kmeans.fit(data)

            for idx in range(len(kmeans.labels_)):
                label = label_start + kmeans.labels_[idx]
                clusters[label].append(indexes[idx])

            label_start += max(kmeans.labels_) + 1

        return clusters

    def approx_hierarchical(self, brand_clusters):
        output_clusters, max_cluster_id = defaultdict(), 0

        for brand, indexes in brand_clusters.items():
            data = self.representations[indexes,:]
            similarities = cosine_similarity(data, data)

            thres_match = similarities >= self.confidence
            pt_indexes = np.sum(thres_match.astype(int), axis=1).argsort()[::-1]

            for idx in pt_indexes:
                q_idx, cluster_id = indexes[idx], max_cluster_id

                if q_idx not in output_clusters:
                    rep = similarities[idx]
                    most_similar = list(zip(rep, range(len(rep)))) if np.sum(rep) > 0 else []

                    output_clusters[q_idx] = (cluster_id, 1.0, q_idx)

                    if len(most_similar) > 0:
                        most_similar = [(x, indexes[y]) for x, y in most_similar if indexes[y] != q_idx and x >= self.confidence]

                        for sim, index in most_similar:
                            if index not in output_clusters or (index in output_clusters and output_clusters[index][1] < sim):
                                output_clusters[index] = (cluster_id, sim, q_idx)

                    max_cluster_id += 1

        clusters = defaultdict(list)

        for pt, cluster in output_clusters.items():
            clusters[cluster[2]].append(pt)

        return clusters
    
    def auto_grouping(self):
        item_text = utils.get_text_data(self.items)
        
        if self.representations is None:
            tfidf_vectorizer = TfidfVectorizer(tokenizer=utils.get_tokens, ngram_range=(1,1), stop_words='english', binary=True)
            self.representations = tfidf_vectorizer.fit_transform(item_text)
            
        brand_clusters = self.cluster_on_brands()
        
        return self.approx_hierarchical(brand_clusters)
        
    def init_groups(self):
        print "Getting actual groups..."
        self.true_groups = self.true_grouping()
        
        print "Getting auto groups..."
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