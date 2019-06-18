import numpy as np

class Solution(object):
    def matrixScore(self, A):
        A = np.array(A)
        curr_row_sums = np.array([0]*len(A))
        
        for col in range(len(A[0])):
            if col == 0:
                for row in range(len(A)):
                    if A[row,col] == 0:
                        A[row] = 1-A[row]
                curr_row_sums = np.array([1]*len(A))
                
            else:
                a = 2*curr_row_sums + A[:,col]
                b = 2*curr_row_sums + 1 - A[:,col]

                c, d = np.sum(a), np.sum(b)

                if d > c:
                    A[:,col] = 1-A[:,col]
                    curr_row_sums = b
                else:
                    curr_row_sums = a
                    
        return np.sum(curr_row_sums)
