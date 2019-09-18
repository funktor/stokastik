class Solution(object):
    def networkDelayTime(self, times, N, K):
        K -= 1
        edge_dict = {}
        for u, v, w in times:
            if u-1 not in edge_dict:
                edge_dict[u-1] = []
            edge_dict[u-1].append((v-1, w))
            
        heap = [(0, K)]
        cache = [float("Inf")]*N
        cache[K] = 0
        
        while len(heap) > 0:
            timestamp, start = heapq.heappop(heap)
            
            if start in edge_dict:
                for x, w in edge_dict[start]:
                    if cache[x] > timestamp+w:
                        heapq.heappush(heap, (timestamp+w, x))
                        cache[x] = timestamp+w
        
        return max(cache) if float("Inf") not in set(cache) else -1
            
            
