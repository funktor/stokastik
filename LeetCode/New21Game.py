class Solution(object):
    def new21Game(self, N, K, W):
        f = [0.0]*(N+1)
        last_sum = 0.0
        
        for k in range(N+1):
            if k <= N-K:
                f[k] = 1.0
            else:
                f[k] = (1.0/W)*last_sum
                
            last_sum = last_sum + f[k] - f[k-W] if k-W >= 0 else last_sum + f[k]
                
        return f[-1]
