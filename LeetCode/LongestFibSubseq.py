class Solution(object):
    def lenLongestFibSubseq(self, A):
        cache = {}
        max_len = -1
        for i in reversed(range(len(A))):
            cache[A[i]] = {}
            for j in range(i, len(A)):
                if j == i:
                    cache[A[i]][A[j]] = 1
                else:
                    if A[i]+A[j] in cache[A[j]]:
                        cache[A[i]][A[j]] = 1 + cache[A[j]][A[i]+A[j]]
                    else:
                        cache[A[i]][A[j]] = 2
                
                max_len = max(max_len, cache[A[i]][A[j]])
        
        return max_len if max_len >= 3 else 0
