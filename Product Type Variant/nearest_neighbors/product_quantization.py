import numpy as np
from scipy.cluster.vq import vq, kmeans2
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from multiprocessing.dummy import Pool as ThreadPool

def get_kmeans_clusters(vectors, num_clusters, use_mini_batch=True):
    if use_mini_batch:
        batch_size = int(min(num_clusters/3.0+1, vectors.shape[0]))
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, init='k-means++')
        kmeans.fit(vectors)
        return kmeans.cluster_centers_, kmeans.labels_
    
    else:
        centroids, labels = kmeans2(vectors, num_clusters, minit='points')
        return centroids, labels
    

class PQ(object):
    def __init__(self, num_partitions, num_codewords_per_partition):
        self.n, self.m = 0, 0
        self.num_partitions = num_partitions
        self.num_codewords_per_partition = num_codewords_per_partition
        self.pqcode = None
        self.codewords = None
        
    def construct(self, vectors):
        self.n, self.m = vectors.shape
        parition_dim = int(self.m / self.num_partitions)
        
        self.codewords = np.empty((self.num_partitions, self.num_codewords_per_partition, parition_dim), np.float32)
        self.pqcode = np.empty((self.n, self.num_partitions), np.uint8)
        
        for m in range(self.num_partitions):
            sub_vectors = vectors[:,m * parition_dim : (m + 1) * parition_dim]
            if sub_vectors.shape[0] == 1:
                self.codewords[m], label = np.mean(sub_vectors, axis=1), np.array([0]*sub_vectors.shape[0])
            else:
                self.codewords[m], label = get_kmeans_clusters(sub_vectors, self.num_codewords_per_partition, use_mini_batch=False)
                
            self.pqcode[:, m], dist = vq(sub_vectors, self.codewords[m])
            
            
    def query_count(self, query, k=5):
        parition_dim = int(self.m / self.num_partitions)
        dist_table = np.empty((self.num_partitions, self.num_codewords_per_partition), np.float32)
        
        for m in range(self.num_partitions):
            query_sub = query[m * parition_dim : (m + 1) * parition_dim]
            dist_table[m, :] = cdist([query_sub], self.codewords[m], 'sqeuclidean')[0]
            
        dist = np.sqrt(np.sum(dist_table[range(self.num_partitions), self.pqcode], axis=1))
        dist = zip(dist, range(self.n))
        dist = sorted(dist, key=lambda k:k[0])
        
        return dist[:min(k, len(dist))]
    
    
    def query_radius(self, query, radius=0.1):
        parition_dim = int(self.m / self.num_partitions)
        dist_table = np.empty((self.num_partitions, self.num_codewords_per_partition), np.float32)
        
        for m in range(self.num_partitions):
            query_sub = query[m * parition_dim : (m + 1) * parition_dim]
            dist_table[m, :] = cdist([query_sub], self.codewords[m], 'sqeuclidean')[0]
            
        dist = np.sqrt(np.sum(dist_table[range(self.num_partitions), self.pqcode], axis=1))
        dist = zip(dist, range(self.n))
        dist = [(x, y) for x, y in dist if x <= radius]
        
        return dist