class Solution(object):
    def num_playlists(self, N, L, K, cache):
        m = 10**9 + 7
        
        if L == 0 or N == 0:
            return 0
        
        if N == L:
            p = 1
            for i in range(1, N+1):
                p *= i % m
                
            cache[(N, L)] = p % m
            return cache[(N, L)]
        
        a = self.num_playlists(N, L-1, K, cache) if (N, L-1) not in cache else cache[(N, L-1)]
        b = self.num_playlists(N-1, L-1, K, cache) if (N-1, L-1) not in cache else cache[(N-1, L-1)]
        
        c, d = N*(a + b), K*a
        
        cache[(N, L)] = (c - d) % m if c >= d else 0
        
        return cache[(N, L)]
        
        
    def numMusicPlaylists(self, N, L, K):
        return self.num_playlists(N, L, K, {})
