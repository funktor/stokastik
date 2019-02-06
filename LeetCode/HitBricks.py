import collections

class Solution(object):
    def get_connected(self, grid, x, y, n, m, is_connected):
        stack, result = [(x,y)], []
        visited = set([(x, y)])
        
        while len(stack) > 0:
            u, v = stack.pop()
            result.append((u,v))
            
            if u == 0 or ((u,v) in is_connected and is_connected[(u,v)]):
                return True, result
            
            for i in [-1,1]:
                if 0 <= u+i < n and 0 <= v < m and grid[u+i][v] == 1 and (u+i, v) not in visited:
                    stack.append((u+i,v))
                    visited.add((u+i,v))
                    
                if 0 <= u < n and 0 <= v+i < m and grid[u][v+i] == 1 and (u, v+i) not in visited:
                    stack.append((u,v+i))
                    visited.add((u,v+i))
        
        return False, result
    
    
    def get_deleted_neighbors(self, grid, x, y, n, m, is_connected):
        stack, result = [(x,y)], []
        visited = set([(x, y)])
        
        while len(stack) > 0:
            u, v = stack.pop()
            result.append((u,v))
            
            for i in [-1,1]:
                if 0 <= u+i < n and 0 <= v < m and (grid[u+i][v] == 2 or (grid[u+i][v] == 1 and is_connected[(u+i,v)] is False)) and (u+i, v) not in visited:
                    stack.append((u+i,v))
                    visited.add((u+i,v))
                    
                if 0 <= u < n and 0 <= v+i < m and (grid[u][v+i] == 2 or (grid[u][v+i] == 1 and is_connected[(u,v+i)] is False)) and (u, v+i) not in visited:
                    stack.append((u,v+i))
                    visited.add((u,v+i))
        
        return result
    
    
    def drop_bricks(self, grid, x, y, n, m, is_connected):
        if 0 <= x < n and 0 <= y < m and grid[x][y] == 1:
            connected, result = self.get_connected(grid, x, y, n, m, is_connected)
            for x, y in result:
                if connected is False:
                    grid[x][y] = 2
                is_connected[(x,y)] = connected
    
            
    def hitBricks(self, grid, hits):
        n, m = len(grid), len(grid[0])
        is_connected = dict()
        
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == 1:
                    is_connected[(x,y)] = False
                    
        for x, y in hits:
            if grid[x][y] == 1:
                grid[x][y] = -1
        
        for x, y in hits:
            if grid[x][y] == -1:
                self.drop_bricks(grid, x+1, y, n, m, is_connected)
                self.drop_bricks(grid, x-1, y, n, m, is_connected)
                self.drop_bricks(grid, x, y+1, n, m, is_connected)
                self.drop_bricks(grid, x, y-1, n, m, is_connected)
                
        out, disconnected_hits = [0]*len(hits), []
        
        for i in reversed(range(len(hits))):
            x, y = hits[i]
            if grid[x][y] == -1:
                grid[x][y] = 1
                
                connected, _ = self.get_connected(grid, x, y, n, m, is_connected)
                is_connected[(x,y)] = connected
                
                if connected:
                    deleted_neighbors = self.get_deleted_neighbors(grid, x, y, n, m, is_connected)
                    cnts = len(deleted_neighbors)-1

                    for u, v in deleted_neighbors:
                        grid[u][v] = 1
                        is_connected[(u,v)] = connected

                    out[i] = cnts
                else:
                    disconnected_hits.append(i)
                    
        for i in disconnected_hits:
            x, y = hits[i]
            if is_connected[(x,y)] is False:
                grid[x][y] = 1
                
                deleted_neighbors = self.get_deleted_neighbors(grid, x, y, n, m, is_connected)
                cnts = len(deleted_neighbors)-1

                out[i] = cnts
                    
        return out
