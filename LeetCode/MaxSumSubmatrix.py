import numpy as np

class Solution(object):
    def maxSumSubmatrix(self, matrix, k):
        matrix = np.transpose(matrix)
        
        count, n, m = 0, len(matrix), len(matrix[0])
        max_sum = -float("Inf")
        
        for i in range(n):
            if i == 0:
                matrix = np.cumsum(np.cumsum(matrix, axis=1), axis=0)
            else:
                matrix = matrix - matrix[0]
                matrix = matrix[1:]
                
            p = np.sum(matrix <= k)
            if p > 0:
                max_sum = max(max_sum, np.max(matrix[matrix <= k]))
            
            for j in range(len(matrix)):
                arr = np.copy(matrix[j])
                mat = np.subtract.outer(arr[::-1], arr)
                if np.max(mat) > max_sum: 
                    c = np.where(mat <= k)
                    d = c[0] + c[1] < len(mat)-1
                    if np.sum(d) > 0:
                        max_sum = max(max_sum, np.max(mat[c[0][d], c[1][d]]))
            
        return max_sum
