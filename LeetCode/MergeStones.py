import numpy as np, collections

class Solution(object):
    def mergeStones(self, stones, K):
        n = len(stones)
        if (n-1) % (K-1) != 0:
            return -1
        
        cached = collections.defaultdict(dict)
        
        for length in range(1, n+1):
            for i in range(n-length+1):
                j = i + length - 1
                if length < K:
                    cached[i][j] = (np.sum(stones[i:j+1]), 0, length)
                elif length == K:
                    sums = np.sum(stones[i:j+1])
                    cached[i][j] = (sums, sums, 1)
                else:
                    min_cost = float("Inf")
                    min_cost_len, min_cost_sum = -1, 0
                    
                    for k in range(i, j):
                        a, b = cached[i][k], cached[k+1][j]
                        if a[2] + b[2] < K:
                            cost = a[1] + b[1]
                            if cost < min_cost:
                                min_cost = cost
                                min_cost_sum = a[0] + b[0]
                                min_cost_len = a[2] + b[2]
                                
                        elif a[2] + b[2] == K:
                            cost = a[1] + b[1] + a[0] + b[0]
                            if cost < min_cost:
                                min_cost = cost
                                min_cost_sum = a[0] + b[0]
                                min_cost_len = 1
                                
                    cached[i][j] = (min_cost_sum, min_cost, min_cost_len)
        
        return cached[0][n-1][1]
