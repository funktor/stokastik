class Solution(object):
    def shortestPathLength(self, graph):
        n = len(graph)
        
        heap = [(0, -1, [])]
        cache = {}
        
        min_path = float("Inf")
        
        while len(heap) > 0:
            cost, node, path = heapq.heappop(heap)
            
            if len(set(path)) == n:
                min_path = min(min_path, len(path)-1)
                if min_path == n-1:
                    return min_path
            
            elif node == -1:
                for i in range(n):
                    heapq.heappush(heap, (-1, i, [i]))
                    cache[((i,), tuple(set([i])))] = 1
            else:
                for i in graph[node]:
                    q = path + [i]
                    if len(q) <= min_path:
                        s = (path[-1], i)
                        p = set(q)

                        if (s, tuple(p)) not in cache or len(q) < cache[(s, tuple(p))]:
                            heapq.heappush(heap, (-len(p), i, q))
                            cache[(s, tuple(p))] = len(q)
                        
        return min_path
