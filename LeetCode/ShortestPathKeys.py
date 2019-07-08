class Solution(object):
    def move(self, grid, x, y, heap, cache, visited, p, steps, min_steps):
        locks = ['A', 'B', 'C', 'D', 'E', 'F']
        keys = ['a', 'b', 'c', 'd', 'e', 'f']
        
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != '#' and steps <= min_steps:
            s = (visited[-1][-2], visited[-1][-1], x, y)
            new_visited = visited + [s]
            
            p1 = p.copy()
                    
            if grid[x][y] in keys:
                p1.add(grid[x][y])
                
            p2 = (s, tuple(p1))
            
            if p2 not in cache or steps+1 < cache[p2]:
                if grid[x][y] not in locks or chr(ord(grid[x][y])+32) in p:
                    heapq.heappush(heap, (-len(p1)+steps+1, x, y, new_visited, p1, steps+1))
                    cache[p2] = steps+1
                
    def shortestPathAllKeys(self, grid):
        num_keys, start_x, start_y = 0, 0, 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] in ['a', 'b', 'c', 'd', 'e', 'f']:
                    num_keys += 1
                
                if grid[i][j] == '@':
                    start_x, start_y = i, j
                    
        heap = [(0, start_x, start_y, [(start_x, start_y)], set(), 0)]
        cache = {((start_x, start_y), tuple(set())):0}
        
        min_steps = float("Inf")
        
        while len(heap) > 0:
            cost, x, y, visited, p, steps = heapq.heappop(heap)
            
            if len(p) == num_keys:
                min_steps = min(min_steps, steps)
                if min_steps == num_keys:
                    return min_steps
                
            else:
                self.move(grid, x+1, y, heap, cache, visited, p, steps, min_steps)
                self.move(grid, x-1, y, heap, cache, visited, p, steps, min_steps)
                self.move(grid, x, y+1, heap, cache, visited, p, steps, min_steps)
                self.move(grid, x, y-1, heap, cache, visited, p, steps, min_steps)
        
        return min_steps if min_steps != float("Inf") else -1
