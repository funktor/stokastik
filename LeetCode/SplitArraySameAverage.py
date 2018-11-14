import numpy as np
import collections

class Solution(object):
    def gcd(self, a, b):
        if a == 0:
            return b
        else:
            return self.gcd(b%a, a)
        
    def get_smaller(self, A, K):
        left, right = 0, len(A)-1

        while left <= right:
            mid = (left+right)/2

            if A[mid] <= K and ((mid < len(A)-1 and A[mid + 1] > K) or mid == len(A)-1):
                return mid
            elif A[mid] > K:
                right = mid - 1
            else:
                left = mid + 1

        return -1
    
    def search_sum(self, A, counts, sums, K):
        if K == 1:
            return sums in set(A)
        else:
            smaller_idx = self.get_smaller(A, sums)
            
            out = False
            for i in reversed(range(smaller_idx+1)):
                for j in range(1, counts[A[i]]+1):
                    new_sums = sums - A[i]*j
                    new_K = K - j
                    
                    if new_sums == 0 and new_K == 0:
                        return True

                    out = out or self.search_sum(A[:i], counts, new_sums, new_K)
                
            return out
        
    def splitArraySameAverage(self, A):
        n = len(A)
        
        if n == 1:
            return False
        
        counts = collections.defaultdict(int)
        
        for x in A:
            counts[x] += 1
            
        B = sorted(set(A))
        
        sums = np.sum(A)
        
        g = self.gcd(sums, n)
        
        a, b = sums/g, n/g
        
        if a == sums:
            return False
        
        for mult in range(1, g):
            a1, b1 = mult*a, mult*b
            
            if self.search_sum(B, counts, a1, b1):
                return True

        return False