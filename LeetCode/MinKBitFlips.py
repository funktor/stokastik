import numpy as np

class Solution(object):
    def minKBitFlips(self, A, K):
        A = np.array(A)
        target = np.array([1]*len(A))
        cnt = 0
        i = len(A)-1
        
        while True:
            while i >= 0 and A[i] == target[i]:
                i -= 1
            
            if i == -1:
                return cnt
            
            if i == K-1:
                return cnt+1 if np.sum(A[:K] == 1 - target[:K]) == K else -1
            
            if i < K-1:
                return -1
            
            target[i+1-K:i+1] = 1-target[i+1-K:i+1]
            cnt += 1
        
        return -1
