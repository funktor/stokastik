class Solution(object):
    def new21Game(self, N, K, W):
        f, g = [0.0]*(N+1), [0.0]*(N+1)
        
        for k in range(N+1):
            if k <= N-K:
                f[k] = 1.0
            else:
                f[k] = (1.0/W)*g[k-1]
                
            g[k] = g[k-1] + f[k] - f[k-W] if k-W >= 0 else g[k-1] + f[k]
                
        return f[-1]
