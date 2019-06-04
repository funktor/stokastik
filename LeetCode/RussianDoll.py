class Solution(object):
    def bsearch(self, num_envelopes, min_num, max_num, k):
        left, right = min_num, max_num
        
        while left <= right:
            mid = (left + right)/2
            
            if num_envelopes[mid] > k:
                left = mid + 1
            else:
                right = mid - 1
                
        return left-1 if left > 0 else left
    
    def bfs(self, envelopes, start, cache, num_envelopes, min_num, max_num):
        if cache[start] == -1:
            a, b = envelopes[start]
            max_depth = 0
            
            d = self.bsearch(num_envelopes, min_num, max_num, envelopes[start][1])
            
            cache[start] = d + 1
            num_envelopes[d + 1] = max(num_envelopes[d + 1], envelopes[start][1])
            
        return cache[start]
            
                    
    def maxEnvelopes(self, envelopes):
        if len(envelopes) == 0:
            return 0
        
        if len(envelopes) == 1:
            return 1
        
        envelopes = sorted(envelopes, key=lambda k:(-k[0], k[1]))
        
        cache = [-1]*len(envelopes)
        num_envelopes = [-1]*(len(envelopes)+1)
        
        min_num, max_num = 1, 1
        
        for i in range(len(envelopes)):
            d = self.bfs(envelopes, i, cache, num_envelopes, min_num, max_num)
            min_num = min(min_num, d)
            max_num = max(max_num, d)
        
        return max_num
