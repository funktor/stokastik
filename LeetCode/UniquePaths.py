import numpy as np

class Solution(object):
    def can_visit(self, grid, x, y, visited):
        return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != -1 and (x, y) not in visited
            
    def sum_paths(self, grid, x, y, visited, num_zeros):
        visited.add((x, y))
        
        a = self.can_visit(grid, x+1, y, visited)
        b = self.can_visit(grid, x-1, y, visited)
        c = self.can_visit(grid, x, y+1, visited)
        d = self.can_visit(grid, x, y-1, visited)
        
        h = num_zeros-1 if grid[x][y] == 0 else num_zeros
        
        flag = False
        
        p, q, r, s = 0, 0, 0, 0
        
        if a and grid[x+1][y] == 0:
            flag = True
            new_visited = visited.copy()
            p = self.sum_paths(grid, x+1, y, new_visited, h)
            
        if b and grid[x-1][y] == 0:
            flag = True
            new_visited = visited.copy()
            q = self.sum_paths(grid, x-1, y, new_visited, h)
            
        if c and grid[x][y+1] == 0:
            flag = True
            new_visited = visited.copy()
            r = self.sum_paths(grid, x, y+1, new_visited, h)
            
        if d and grid[x][y-1] == 0:
            flag = True
            new_visited = visited.copy()
            s = self.sum_paths(grid, x, y-1, new_visited, h)
            
        if flag:
            return p + q + r + s
        else:
            if (a or b or c or d) and h == 0:
                return 1
            return 0
        
    
    def uniquePathsIII(self, grid):
        x, y = np.argwhere(np.array(grid) == 1)[0]
        n, m = len(grid), len(grid[0])
        num_zeros = n*m - np.count_nonzero(grid)
                    
        return self.sum_paths(grid, x, y, set(), num_zeros)
        
