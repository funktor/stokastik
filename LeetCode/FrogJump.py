class Solution(object):
    def bsearch(self, stones, start, end, k):
        left, right = start, end
        while left <= right:
            mid = (left + right)/2
            if stones[mid] == k:
                return mid
            elif stones[mid] < k:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    def cross(self, stones, i, k, cached):
        if i == len(stones)-1:
            return True
        
        a = self.bsearch(stones, i, len(stones)-1, stones[i] + k-1)
        b = self.bsearch(stones, i, len(stones)-1, stones[i] + k)
        c = self.bsearch(stones, i, len(stones)-1, stones[i] + k+1)
        
        if a != -1 and a != i:
            p = self.cross(stones, a, k-1, cached) if (a, k-1) not in cached else cached[(a, k-1)]
        else:
            p = False
            
        if b != -1 and b != i:
            q = self.cross(stones, b, k, cached) if (b, k) not in cached else cached[(b, k)]
        else:
            q = False
            
        if c != -1 and c != i:
            r = self.cross(stones, c, k+1, cached) if (c, k+1) not in cached else cached[(c, k+1)]
        else:
            r = False
            
        cached[(i, k)] = p or q or r
        return p or q or r
    
    def canCross(self, stones):
        if stones[1] != 1:
            return False
        
        return self.cross(stones, 1, 1, {})
