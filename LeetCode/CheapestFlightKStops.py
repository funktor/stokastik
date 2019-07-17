class Solution(object):
    def cheapest(self, adj, src, dst, K, cache, visited):
        if src not in adj:
            return float("Inf")
        
        elif K == 0:
            return adj[src][dst] if dst in adj[src] else float("Inf")
        
        else:
            min_cost = float("Inf")
            visited.add(src)
            
            for x in adj[src]:
                if x == dst:
                    min_cost = min(min_cost, adj[src][dst])
                else:
                    if x not in visited:
                        new_visited = visited.copy()
                        a = self.cheapest(adj, x, dst, K-1, cache, new_visited) if (x, K-1) not in cache else cache[(x, K-1)]
                        min_cost = min(min_cost, adj[src][x] + a)
            
            cache[(src, K)] = min_cost
            return min_cost
        
    def findCheapestPrice(self, n, flights, src, dst, K):
        adj = {}
        for s,d,p in flights:
            if s not in adj:
                adj[s] = {}
            adj[s][d] = p
        
        out = self.cheapest(adj, src, dst, K, {}, set())
        return out if out != float("Inf") else -1
