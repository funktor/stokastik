import collections

class Solution(object):
    def distance(self, grid, i, j, dist_cache):
        heap = [(0, i, j, 0)]
        
        while len(heap) > 0:
            cost, i, j, dist = heapq.heappop(heap)
            
            for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                    if grid[x][y] == 1:
                        if (x, y) not in dist_cache:
                            heapq.heappush(heap, (0, x, y, 0))
                            dist_cache[(x, y)] = 0
                    else:
                        if (x, y) not in dist_cache or dist_cache[(x, y)] > dist+1:
                            heapq.heappush(heap, (dist+1, x, y, dist+1))
                            dist_cache[(x, y)] = dist+1
                
                    
    def maxDistance(self, grid):
        dist_cache = {}
        max_dist = -1
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    self.distance(grid, i, j, dist_cache)
        
                    max_dist = -1
                    for k, v in dist_cache.items():
                        max_dist = max(max_dist, v)

                    return max_dist if max_dist > 0 else -1
        return -1
        
