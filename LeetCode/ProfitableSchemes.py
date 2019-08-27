class Solution(object):
    def get_max(self, profit_sums, p):
        left, right = 0, len(profit_sums)-1
        while left <= right:
            mid = (left + right)/2
            if profit_sums[mid] < p:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    def profitableSchemes(self, G, P, group, profit):
        profit_sums = [0]*len(profit)
        for i in range(len(profit)):
            if i == 0:
                profit_sums[i] = profit[i]
            else:
                profit_sums[i] = profit[i] + profit_sums[i-1]
                
        m = 10**9+7
        
        cache = [[[0]*len(group) for j in range(P+1)] for i in range(G+1)]
        
        for g in range(1, G+1):
            for p in range(P+1):
                ind = self.get_max(profit_sums, p)
                
                for i in range(ind, len(group)):
                    if i == 0:
                        cache[g][p][i] = 1 if profit[i] >= p and group[i] <= g else 0
                    else:
                        a = cache[g-group[i]][max(0, p-profit[i])][i-1] if g >= group[i]+1 else 0
                        b = cache[g][p][i-1]
                        c = 1 if profit[i] >= p and group[i] <= g else 0
                        cache[g][p][i] = (a+b+c) % m
        
        return cache[G][P][len(group)-1]
