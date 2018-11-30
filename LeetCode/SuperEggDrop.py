class Solution(object):
    def get_egg_solution(self, K, N, cache):
        if K == 1:
            return N
        elif N == 0:
            return 0
        else:
            left, right = 1, N
            while left < right:
                mid = (left + right)/2
                
                if (K-1, mid-1) in cache:
                    a = cache[(K-1, mid-1)]
                else:
                    a = self.get_egg_solution(K-1, mid-1, cache)
                    
                if (K, N-mid) in cache:
                    b = cache[(K, N-mid)]
                else:
                    b = self.get_egg_solution(K, N-mid, cache)
                
                if a >= b:
                    right = mid
                else:
                    left = mid + 1
            
            if (K-1, left-1) in cache:
                a = cache[(K-1, left-1)]
            else:
                a = self.get_egg_solution(K-1, left-1, cache)

            if (K, N-left) in cache:
                b = cache[(K, N-left)]
            else:
                b = self.get_egg_solution(K, N-left, cache)
            
            out = 1 + max(a, b)
            cache[(K, N)] = out
            
            return out
        
    def superEggDrop(self, K, N):
        cache = dict()
        return self.get_egg_solution(K, N, cache)