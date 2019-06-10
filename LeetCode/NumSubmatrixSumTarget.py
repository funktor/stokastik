import numpy as np

class Solution(object):
    def numSubmatrixSumTarget(self, matrix, target):
        count, n, m = 0, len(matrix), len(matrix[0])
        
        for i in range(n):
            if i == 0:
                matrix = np.cumsum(np.cumsum(matrix, axis=1), axis=0)
            else:
                matrix = matrix - matrix[0]
                matrix = matrix[1:]
            
            count += np.sum(matrix == target)
            
            for j in range(len(matrix)):
                cache = dict()
                for k in range(len(matrix[0])):
                    rem = matrix[j][k] - target
                    if rem in cache:
                        count += cache[rem]
                    
                    if matrix[j][k] not in cache:
                        cache[matrix[j][k]] = 0
                    cache[matrix[j][k]] += 1
        
        return count
