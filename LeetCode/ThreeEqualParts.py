import numpy as np

class Solution(object):
    def threeEqualParts(self, A):
        n, m = len(A), np.sum(A)
        
        if m % 3 != 0:
            return [-1,-1]
        elif m == 0:
            return [0,n-1]
        
        num_ones_each, rep = m/3, []
        
        i = n-1
        while num_ones_each > 0:
            rep.append(A[i])
            if A[i] == 1:
                num_ones_each -= 1
            i -= 1
        
        rep = rep[::-1]
        u, end = len(rep), i+1
        
        i, pos = 0, []
        while i < end:
            if A[i] == 1:
                if A[i:(i+u)] == rep:
                    pos.append(i+u-1)
                    i += u
                else:
                    return [-1,-1]
            else:
                i += 1
        
        return [pos[0], pos[1]+1]