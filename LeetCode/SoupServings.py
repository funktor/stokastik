class Solution(object):
    def soup(self, X, Y, cache):
        if X <= 0:
            return (1.0, 0.0) if Y > 0 else (0.0, 1.0)
        
        if Y <= 0:
            return (0.0, 0.0)
        
        a1, a2 = self.soup(X-100, Y, cache) if (X-100, Y) not in cache else cache[(X-100, Y)]
        b1, b2 = self.soup(X-75, Y-25, cache) if (X-75, Y-25) not in cache else cache[(X-75, Y-25)]
        c1, c2 = self.soup(X-50, Y-50, cache) if (X-50, Y-50) not in cache else cache[(X-50, Y-50)]
        d1, d2 = self.soup(X-25, Y-75, cache) if (X-25, Y-75) not in cache else cache[(X-25, Y-75)]
        
        cache[(X, Y)] = (0.25*(a1 + b1 + c1 + d1), 0.25*(a2 + b2 + c2 + d2))
        
        return cache[(X, Y)]
    
    def soupServings(self, N):
        if N >= 5000:
            return 1.0
        
        a, b = self.soup(N, N, {})
        return a + 0.5*b
