import math
class Solution(object):
    def consecutiveNumbersSum2(self, N):
        m = int(math.sqrt(2*N))
        cnt = 0
        for i in range(1, m+1):
            u = 0.5*i*(i-1)
            if N > u and (N-u) % i == 0:
                cnt += 1
        return cnt
    
    def consecutiveNumbersSum(self, N):
        start, m, cnt = N, 1, 0
        while start-m > 0:
            u = start-m
            if u % (m+1) == 0:
                cnt += 1
            start = u
            m += 1
            
        return cnt+1
        
