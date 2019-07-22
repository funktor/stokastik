class Solution(object):
    def game(self, piles, start, end, cache):
        if start == end-1:
            return max(piles[start], piles[end])
        
        a = self.game(piles, start+2, end, cache) if (start+2, end) not in cache else cache[(start+2, end)]
        b = self.game(piles, start+1, end-1, cache) if (start+1, end-1) not in cache else cache[(start+1, end-1)]
        c = self.game(piles, start, end-2, cache) if (start, end-2) not in cache else cache[(start, end-2)]
        
        x, y = piles[start] + min(a, b), piles[end] + min(b, c)
        
        cache[(start, end)] = max(x, y)
        
        return cache[(start, end)]
        
        
    def stoneGame(self, piles):
        out = self.game(piles, 0, len(piles)-1, {})
        return out > 0.5*sum(piles)
            
