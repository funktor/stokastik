class Solution(object):
    def findLength(self, A, B):
        cache = [[0]*len(B) for i in range(len(A))]
        
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i] == B[j]:
                    if i > 0 and j > 0:
                        cache[i][j] = cache[i-1][j-1] + 1
                    else:
                        cache[i][j] = 1
                else:
                    cache[i][j] = 0
        
        return max([max(x) for x in cache])
