import numpy as np

class Solution(object):
    def minDeletionSize(self, A):
        A = np.array([list(x) for x in A])
        cache = [-1]*len(A[0])
        
        max_len = -1
        for i in range(len(A[0])):
            if i == 0:
                cache[i] = 1
            else:
                for j in range(i):
                    a = 1 + cache[j] if np.sum(A[:,i] >= A[:,j]) == len(A) else 1
                    cache[i] = max(cache[i], a)
            
            max_len = max(max_len, cache[i])
        
        return len(A[0]) - max_len
