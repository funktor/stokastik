import collections, numpy as np, time

class Solution(object):
    def reconstruct(self, b_cache, cache, sums, rods, start, half_sums):
        p = b_cache[sums]
        q = cache[sums][start+1] if sums in cache and start+1 in cache[sums] else None
        
        out = False
        
        if q is not None:
            for j in p:
                if j >= q:
                    if half_sums-rods[j] == 0:
                        return True
                    elif half_sums in cache and j in cache[half_sums] and cache[half_sums][j] == j:
                        out = out or self.reconstruct(b_cache, cache, sums-rods[j], rods, j, half_sums-rods[j])
                    else:
                        out = out or self.reconstruct(b_cache, cache, sums-rods[j], rods, j, half_sums)
        return out
                
    def tallestBillboard(self, rods):
        cache, b_cache = collections.defaultdict(dict), collections.defaultdict(list)
        rods = sorted(rods)
        
        n = len(rods)
        
        if n == 0:
            return 0
        
        min_sums, max_sums = np.min(rods), np.sum(rods)
        
        for sums in range(min_sums, max_sums+1):
            for i in reversed(range(n)):
                y = sums-rods[i]
                a, b = sums in cache and i+1 in cache[sums], y in cache and i+1 in cache[y]
                
                if y == 0:
                    cache[sums][i] = i
                    b_cache[sums].append(i)
                elif y > 0:
                    if b:
                        cache[sums][i] = i
                        b_cache[sums].append(i)
                    elif a:
                        cache[sums][i] = cache[sums][i+1]
                else:
                    if a:
                        cache[sums][i] = cache[sums][i+1]
        
        valid_sums = sorted([sums for sums in b_cache])
        
        for sums in reversed(valid_sums):
            if sums % 2 == 0:
                half_sums = sums/2
                
                out = self.reconstruct(b_cache, cache, sums, rods, -1, half_sums)
                if out:
                    return half_sums
        
        return 0
