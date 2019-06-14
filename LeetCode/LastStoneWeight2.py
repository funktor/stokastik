import collections

class Solution(object):
    def smashed(self, stones, weight, i, cached, max_weight):
        if i == len(stones)-1:
            cached[weight][i] = stones[i] == weight
        else:
            if abs(stones[i]-weight) in cached and i+1 in cached[abs(stones[i]-weight)]:
                a = cached[abs(stones[i]-weight)][i+1]
            else:
                a = self.smashed(stones, abs(stones[i]-weight), i+1, cached, max_weight)
                
            if stones[i]+weight in cached and i+1 in cached[stones[i]+weight]:
                b = cached[stones[i]+weight][i+1]
            else:
                b = self.smashed(stones, stones[i]+weight, i+1, cached, max_weight)
                
            cached[weight][i] = a or b
        
        return cached[weight][i]
    
    def lastStoneWeightII(self, stones):
        max_weight = max(stones)
        cached = collections.defaultdict(dict)
        
        min_weight = float("Inf")
        
        for weight in range(max_weight+1):
            out = self.smashed(stones, weight, 0, cached, max_weight)
            if out:
                min_weight = min(min_weight, weight)
        
        return min_weight
