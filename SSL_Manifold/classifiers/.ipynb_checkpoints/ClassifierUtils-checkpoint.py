import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import kneighbors_graph, BallTree
import scipy.sparse as sp
import scipy as sc

def compute_KL_matrix(X, n_neighbors, kernel, l, u, L_normalized):
    ball_tree = BallTree(X, leaf_size=50)
        
    print('Computing adjacent matrix')
    W = kneighbors_graph(ball_tree, n_neighbors, mode='distance', metric='cosine')
    print('done')

    print('Computing laplacian graph')
    if L_normalized:
        L = normalized_laplacian(W)
    else:
        L = np.diag(W.sum(axis=1)) - W
    print('done')

    print('Computing kernel matrix')
    K = kernel(X)
    print('done')
    
    return K, L
    

def compute_thresholds(K, alpha, Y_in, l, u, search_space_size=201):
    print('Learning threshold')
            
    preds = K.dot(alpha)
    search_space_min = np.amin(preds, axis=0).tolist()[0]
    search_space_max = np.amax(preds, axis=0).tolist()[0]

    def to_minimize(i, threshold):
        predictions = np.array((preds[:,i] > threshold) * 1)
        predictions[predictions == 0] = -1
        p, r, f, s = precision_recall_fscore_support(Y_in[:l, i], predictions[:l])
        return -np.sum(p*s)/np.sum(s)

    search_space = [np.linspace(x, y, num=search_space_size) for x, y in zip(search_space_min, search_space_max)]
    thresholds = [0.0]*Y_in.shape[1]

    for i in range(len(search_space)):
        res = [to_minimize(i, j) for j in search_space[i]]
        thresholds[i] = np.mean(search_space[i][res == np.min(res)])
    
    return thresholds


def normalized_laplacian(W):
    n, m = W.shape
    diag = np.diag(W.sum(axis=0))
    with sc.errstate(divide='ignore'):
        diags_sqrt = 1.0 / sc.sqrt(diag)
    diags_sqrt[sc.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], m, n, format='csr')
    DH=DH.toarray()
    L=DH.dot(L.dot(DH))
    
    return L