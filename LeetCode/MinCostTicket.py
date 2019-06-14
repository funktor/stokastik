class Solution(object):
    def bsearch(self, days, k):
        left, right = 0, len(days)-1
        while left <= right:
            mid = (left + right)/2
            if days[mid] == k:
                return mid
            elif days[mid] > k:
                right = mid - 1
            else:
                left = mid + 1
        return right + 1
    
    def min_cost(self, days, costs, i, cached):
        i1 = self.bsearch(days, i+1)
        i2 = self.bsearch(days, i+7)
        i3 = self.bsearch(days, i+30)
        
        if i1 < len(days):
            if days[i1] in cached:
                a = costs[0] + cached[days[i1]]
            else:
                a = costs[0] + self.min_cost(days, costs, days[i1], cached)
        else:
            a = costs[0]
            
        if i2 < len(days):
            if days[i2] in cached:
                b = costs[1] + cached[days[i2]]
            else:
                b = costs[1] + self.min_cost(days, costs, days[i2], cached)
        else:
            b = costs[1]
            
        if i3 < len(days):
            if days[i3] in cached:
                c = costs[2] + cached[days[i3]]
            else:
                c = costs[2] + self.min_cost(days, costs, days[i3], cached)
        else:
            c = costs[2]
        
        cached[i] = min(a, b, c)
        return cached[i]
            
    def mincostTickets(self, days, costs):
        return self.min_cost(days, costs, days[0], {})
