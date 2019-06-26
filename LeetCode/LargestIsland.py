import collections

class Solution(object):
    def island(self, grid, start, all_cache, inv_map):
        queue = collections.deque([start])
        visited = set([start])
        
        while len(queue) > 0:
            x, y = queue.popleft()
            
            if x+1 < len(grid) and (x+1, y) not in visited and grid[x+1][y] == 1:
                if (x+1, y) in all_cache:
                    ids = all_cache[(x+1, y)]
                    visited.update(inv_map[ids])
                else:
                    queue.append((x+1, y))
                    visited.add((x+1, y))
                
            if x-1 >= 0 and (x-1, y) not in visited and grid[x-1][y] == 1:
                if (x-1, y) in all_cache:
                    ids = all_cache[(x-1, y)]
                    visited.update(inv_map[ids])
                else:
                    queue.append((x-1, y))
                    visited.add((x-1, y))
                
            if y+1 < len(grid[0]) and (x, y+1) not in visited and grid[x][y+1] == 1:
                if (x, y+1) in all_cache:
                    ids = all_cache[(x, y+1)]
                    visited.update(inv_map[ids])
                else:
                    queue.append((x, y+1))
                    visited.add((x, y+1))
                
            if y-1 >= 0 and (x, y-1) not in visited and grid[x][y-1] == 1:
                if (x, y-1) in all_cache:
                    ids = all_cache[(x, y-1)]
                    visited.update(inv_map[ids])
                else:
                    queue.append((x, y-1))
                    visited.add((x, y-1))
        
        return visited
            
        
    def largestIsland(self, grid):
        all_cache, ids, inv_map = {}, 0, {}
        max_island = -1
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1 and (i, j) not in all_cache:
                    visited = self.island(grid, (i, j), {}, {})
                    max_island = max(max_island, len(visited))
                    for x, y in visited:
                        all_cache[(x, y)] = ids
                    
                    inv_map[ids] = visited
                    ids += 1
        
        # print inv_map
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    grid[i][j] = 1
                    visited = self.island(grid, (i, j), all_cache, inv_map)
                    max_island = max(max_island, len(visited))
                    grid[i][j] = 0
        
        return max_island
