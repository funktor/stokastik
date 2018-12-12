class Solution(object):
    def sumSubseqWidths(self, A):
        A = sorted(A)
        n, m = len(A), 10**9+7
        
        sums, last_pos, last_neg = 0, 1, 1
        
        for i in range(len(A)):
            if i == 0:
                pos, neg = 2**i, 2**(n-i-1)
            else:
                pos, neg = last_pos*2, last_neg/2
            
            sums += (A[i] * (pos - neg))%m
            last_pos, last_neg = pos, neg
        
        return sums%m
