class Solution(object):
    def knight(self, N, K, r, c, cache):
        if K == 0:
            return 1.0
        
        w = [(1,2), (1,-2), (2,1), (2,-1), (-1,2), (-1,-2), (-2,1), (-2,-1)]
        x = 0
        for p, q in w:
            if 0 <= r+p < N and 0 <= c+q < N:
                x += self.knight(N, K-1, r+p, c+q, cache) if (K-1, r+p, c+q) not in cache else cache[(K-1, r+p, c+q)]
        
        cache[(K, r, c)] = 0.125*x
        return cache[(K, r, c)]
            
            
    def knightProbability(self, N, K, r, c):
        return self.knight(N, K, r, c, {})
        
