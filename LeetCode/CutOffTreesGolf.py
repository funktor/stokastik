from collections import deque
import heapq, numpy as np

class Solution(object):
    def add(self, forest, x, y, stack, stack_pos):
        if 0 <= x < len(forest) and 0 <= y < len(forest[0]) and forest[x][y] != 0:
            stack.append((x, y))
            stack_pos[(x, y)] = len(stack)-1
            return True
        return False
    
    
    def get_add_status(self, forest, end, x, y, stack, stack_pos, p=1, q=1):
        if end[0] == x:
            z = self.add(forest, x, y+q, stack, stack_pos)
        elif end[1] == y:
            z = self.add(forest, x+p, y, stack, stack_pos)
        else:
            z = self.add(forest, x+p, y, stack, stack_pos) or self.add(forest, x, y+q, stack, stack_pos)
        return z
    
    
    def dfs_update(self, forest, stack, stack_pos, end):
        curr_start = stack[-1]
        x, y = curr_start

        z = False
        
        if end[0] >= x and end[1] >= y:
            z = self.get_add_status(forest, end, x, y, stack, stack_pos, 1, 1)
            
        elif end[0] >= x and end[1] <= y:
            z = self.get_add_status(forest, end, x, y, stack, stack_pos, 1, -1)
            
        elif end[0] <= x and end[1] >= y:
            z = self.get_add_status(forest, end, x, y, stack, stack_pos, -1, 1)

        else:
            z = self.get_add_status(forest, end, x, y, stack, stack_pos, -1, -1)

        return z
            
            
    def dfs(self, forest, start, end):
        min_dist = float("Inf")
        
        stack_fwd, stack_bwd = [start], [end]
        stack_fwd_pos, stack_bwd_pos = {start:0}, {end:0}
        
        while True:
            x = self.dfs_update(forest, stack_fwd, stack_fwd_pos, end)
            
            if x is False:
                return -1
            
            if stack_fwd[-1] in stack_bwd_pos:
                return stack_bwd_pos[stack_fwd[-1]] + len(stack_fwd) - 1
                    
            y = self.dfs_update(forest, stack_bwd, stack_bwd_pos, start)
            
            if y is False:
                return -1
                    
            if stack_bwd[-1] in stack_fwd_pos:
                return stack_fwd_pos[stack_bwd[-1]] + len(stack_bwd) - 1
            
            
    def a_star_add(self, forest, end, x, y, dist, visited, min_heap, distances):
        if 0 <= x < len(forest) and 0 <= y < len(forest[0]) and forest[x][y] != 0 and (x, y) not in visited:
            cost = abs(end[0]-x) + abs(end[1]-y) + dist + 1

            if (x, y) not in distances or distances[(x, y)] > dist+1:
                heapq.heappush(min_heap, (cost, (x, y), dist+1))
                distances[(x, y)] = dist+1
            
            
    def a_star(self, forest, start, end):
        if start == end:
            return 0
        
        z = self.dfs(forest, start, end)
            
        if z != -1:
            return z
        
        min_heap = [(abs(end[0]-start[0]) + abs(end[1]-start[1]), start, 0)]
        distances, visited = {start:0}, set()
        
        while len(min_heap) > 0:
            cost, curr_start, dist = heapq.heappop(min_heap)
            visited.add(curr_start)
            
            if curr_start == end:
                return dist
            
            x, y = curr_start
            
            self.a_star_add(forest, end, x+1, y, dist, visited, min_heap, distances)
            self.a_star_add(forest, end, x-1, y, dist, visited, min_heap, distances)
            self.a_star_add(forest, end, x, y+1, dist, visited, min_heap, distances)
            self.a_star_add(forest, end, x, y-1, dist, visited, min_heap, distances)
        
        return -1
        
            
    def cutOffTree(self, forest):
        height_min_heap = []
        
        for i in range(len(forest)):
            for j in range(len(forest[0])):
                if forest[i][j] > 0:
                    height_min_heap.append((forest[i][j], (i, j)))
                    
        heapq.heapify(height_min_heap)
        
        walks, start = 0, (0,0)
        while len(height_min_heap) > 0:
            height, end = heapq.heappop(height_min_heap)
            walk = self.a_star(forest, start, end)
            
            if walk == -1:
                return -1
            walks += walk
            start = end
        
        return walks
