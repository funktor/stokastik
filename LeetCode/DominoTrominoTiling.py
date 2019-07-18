class Solution(object):
    def numTilings(self, N):
        m = 10**9 + 7
        f, g = [0]*(N+1), [0]*(N+1)
        
        for i in range(N+1):
            if i <= 1:
                f[i] = 1
            elif i == 2:
                f[i] = 2
            else:
                f[i] = (f[i-1] + f[i-2] + 2*g[i-3]) % m
            
            g[i] = (g[i-1] + f[i]) % m
        
        return f[N]
