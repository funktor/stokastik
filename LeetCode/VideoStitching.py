class Solution(object):
    def bsearch(self, clips, k, start):
        left, right = start, len(clips)-1
        while left <= right:
            mid = (left + right)/2
            if clips[mid][0] > k:
                right = mid - 1
            else:
                left = mid + 1
        return right if right >= start else -1
        
            
    def videoStitching(self, clips, T):
        clips = sorted(clips, key=lambda k:(k[0], k[1]))
        k, count, start = 0, 1, 0
        while True:
            ind = self.bsearch(clips, k, start)
            if ind == -1:
                return -1
            
            max_h, best = -float("Inf"), -1
            for i in range(start, ind+1):
                if clips[i][1] > max_h:
                    max_h = clips[i][1]
                    best = i
                    
            if max_h >= T:
                return count
            
            count += 1
            k = max_h
            start = best + 1
        
        return count
        
