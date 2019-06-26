class Solution(object):
    def dfs(self, adj_list, start, visited, cache):
        visited.add(start)
        p, q = 0, 0
        
        if start in adj_list:
            for x in adj_list[start]:
                if x not in visited:
                    new_visited = visited.copy()
                    a, b = self.dfs(adj_list, x, new_visited, cache)
                    p += a+b+1
                    q += b+1
        
        cache[start] = (p, q)
        return cache[start]
        
        
    def bfs_sum(self, adj_list, dfs_cache, N):
        queue = collections.deque([(0, dfs_cache[0][0], dfs_cache[0][1])])
        visited = set([0])
        bfs_cache = [-1]*N
        
        while len(queue) > 0:
            q, a, b = queue.popleft()
            bfs_cache[q] = a
            
            for x in adj_list[q]:
                if x not in visited:
                    v = a + b - 2*dfs_cache[x][1] - 1
                    queue.append((x, v, b))
                    visited.add(x)
                    
        return bfs_cache
    
            
    def sumOfDistancesInTree(self, N, edges):
        if len(edges) == 0:
            return [0]
        
        adj_list = {}
        for x, y in edges:
            if x not in adj_list:
                adj_list[x] = []
            adj_list[x].append(y)
            
            if y not in adj_list:
                adj_list[y] = []
            adj_list[y].append(x)
            
        dfs_cache = [-1]*N
        self.dfs(adj_list, 0, set(), dfs_cache)
        
        # print dfs_cache
        bfs_cache = self.bfs_sum(adj_list, dfs_cache, N)
        
        return bfs_cache
