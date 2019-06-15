class Solution(object):
    def num_distinct(self, s, t, i, j, cache):
        c = 0
        
        if i < j:
            return 0
        
        if j == -1:
            return 1
        
        elif s[i] != t[j]:
            if (i-1, j) in cache:
                return cache[(i-1, j)]
            c = self.num_distinct(s, t, i-1, j, cache)
        
        else:
            a = self.num_distinct(s, t, i-1, j, cache) if (i-1, j) not in cache else cache[(i-1, j)]
            b = self.num_distinct(s, t, i-1, j-1, cache) if (i-1, j-1) not in cache else cache[(i-1, j-1)]
            
            c = a + b
        
        cache[(i, j)] = c
        return c
        
    def numDistinct(self, s, t):
        return self.num_distinct(s, t, len(s)-1, len(t)-1, {})
