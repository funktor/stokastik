import numpy as np

class Solution(object):
    def numSubmatrixSumTarget(self, matrix, target):
        mat_sums = np.zeros((len(matrix), len(matrix), len(matrix[0])), dtype="int32")
        cache = dict()
        count = 0
        
        for nrows in range(1, len(matrix)+1):
            for i in range(len(matrix)-nrows+1):
                j = i + nrows - 1
                if (i, j) not in cache:
                    cache[(i, j)] = dict()
                    
                for k in range(len(matrix[0])):
                    if i == j:
                        if k == 0:
                            mat_sums[i][j][k] = matrix[i][k]
                        else:
                            mat_sums[i][j][k] = mat_sums[i][j][k-1] + matrix[i][k]
                    else:
                        mat_sums[i][j][k] = mat_sums[i][j-1][k] + mat_sums[j][j][k]
                        
                    if mat_sums[i][j][k] == target:
                        count += 1
                    
                    rem = mat_sums[i][j][k] - target
                    if rem in cache[(i, j)]:
                        count += cache[(i, j)][rem]
                        
                    if mat_sums[i][j][k] not in cache[(i, j)]:
                        cache[(i, j)][mat_sums[i][j][k]] = 0
                        
                    cache[(i, j)][mat_sums[i][j][k]] += 1
        
        return count
