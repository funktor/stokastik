class Solution(object):
    def longestArithSeqLength(self, A):
        cached = {}
        
        for i in range(len(A)):
            if i not in cached:
                cached[i] = {}
            if i == 0:
                cached[i][float("Inf")] = 1
                
            for j in range(i):
                if A[i]-A[j] in cached[j]:
                    cached[i][A[i]-A[j]] = 1 + cached[j][A[i]-A[j]]
                else:
                    cached[i][A[i]-A[j]] = 1
        
        max_len = -1
        for i in range(len(A)):
            for h, k in cached[i].items():
                max_len = max(max_len, k+1)
        
        return max_len
