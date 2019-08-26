class Solution(object):
    def stone_game(self, piles, piles_sum, start, M, cache):
        if start + 2*M >= len(piles):
            z = piles_sum[-1]-piles_sum[start-1] if start > 0 else piles_sum[-1]
            return z, 0
        
        max_stones, other = -float("Inf"), 0
        
        for i in range(1, 2*M+1):
            new_M = max(M, i)
            x, y = self.stone_game(piles, piles_sum, start+i, new_M, cache) if (start+i, new_M) not in cache else cache[(start+i, new_M)]
            z = piles_sum[start+i-1]-piles_sum[start-1] if start > 0 else piles_sum[start+i-1]
            
            if z + y > max_stones:
                max_stones = z + y
                other = x
        
        cache[(start, M)] = (max_stones, other)
        return cache[(start, M)]
    
    def stoneGameII(self, piles):
        piles_sum = [0]*len(piles)
        for i in range(len(piles)):
            if i == 0:
                piles_sum[i] = piles[i]
            else:
                piles_sum[i] = piles_sum[i-1] + piles[i]
        
        x, y = self.stone_game(piles, piles_sum, 0, 1, {})
        return x
        
