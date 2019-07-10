class Solution(object):
    def get_money(self, start, end, cache):
        if start >= end:
            return 0
        
        min_cost = float("Inf")
        
        for i in range(start, end+1):
            a = self.get_money(start, i-1, cache) if (start, i-1) not in cache else cache[(start, i-1)]
            b = self.get_money(i+1, end, cache) if (i+1, end) not in cache else cache[(i+1, end)]
            
            min_cost = min(min_cost, i + max(a, b))
        
        cache[(start, end)] = min_cost
        
        return min_cost
            
    def getMoneyAmount(self, n):
        return self.get_money(1, n, {})
