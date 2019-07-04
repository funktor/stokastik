class Solution(object):
    def bfs(self, heightMap, i, j, m, n):
        base = heightMap[i][j]
        queue = collections.deque([(i, j)])
        visited = set([(i, j)])
        
        min_value = float("Inf")
        
        while len(queue) > 0:
            x, y = queue.popleft()
            x0 = heightMap[x][y]
            
            if x == 0 or y == 0 or x == m-1 or y == n-1:
                if heightMap[x][y] > base:
                    visited.remove((x, y))
                    min_value = min(min_value, heightMap[x][y])
                else:
                    return set(), float("Inf")
            
            elif heightMap[x][y] > base:
                visited.remove((x, y))
                min_value = min(min_value, heightMap[x][y])
            
            else:
                if (x+1, y) not in visited:
                    queue.append((x+1, y))
                    visited.add((x+1, y))
                    
                if (x-1, y) not in visited:
                    queue.append((x-1, y))
                    visited.add((x-1, y))
                    
                if (x, y+1) not in visited:
                    queue.append((x, y+1))
                    visited.add((x, y+1))
                    
                if (x, y-1) not in visited:
                    queue.append((x, y-1))
                    visited.add((x, y-1))
                    
        return visited, min_value
    
    def fn(self, heightMap, x, y, heap, visited, out_set, h, excluded, m, n):
        p = 0 < x < m-1 and 0 < y < n-1
            
        if p and (x, y) not in visited and (x, y) not in excluded:
            visited.add((x, y))
            
            if heightMap[x][y] >= h:
                out_set.add((x, y))
                heapq.heappush(heap, (heightMap[x][y], x, y))
            else:
                heapq.heappush(heap, (h, x, y))
    
    def bfs2(self, heightMap, i, j, m, n, excluded):
        heap = [(heightMap[i][j], i, j)]
        visited, out_set = set([(i, j)]), set()
        
        while len(heap) > 0:
            h, x, y = heapq.heappop(heap)
            
            self.fn(heightMap, x+1, y, heap, visited, out_set, h, excluded, m, n)
            self.fn(heightMap, x-1, y, heap, visited, out_set, h, excluded, m, n)
            self.fn(heightMap, x, y+1, heap, visited, out_set, h, excluded, m, n)
            self.fn(heightMap, x, y-1, heap, visited, out_set, h, excluded, m, n)
                    
        return out_set
            
    def trapRainWater(self, heightMap):
        if len(heightMap) == 0:
            return 0
        
        m, n = len(heightMap), len(heightMap[0])
        
        excluded = set()
        
        for j in range(1, n-1):
            x = self.bfs2(heightMap, 0, j, m, n, excluded)
            excluded.update(x)
            y = self.bfs2(heightMap, m-1, j, m, n, excluded)
            excluded.update(y)
            
        for i in range(1, m-1):
            x = self.bfs2(heightMap, i, 0, m, n, excluded)
            excluded.update(x)
            y = self.bfs2(heightMap, i, n-1, m, n, excluded)
            excluded.update(y)
            
        heap = []
        
        for i in range(1, m-1):
            for j in range(1, n-1):
                if (i, j) not in excluded:
                    heap.append((-heightMap[i][j], i, j))
        
        heapq.heapify(heap)
        total_sum, visited = 0, set()
        
        while len(heap) > 0:
            base, i, j = heapq.heappop(heap)
            
            if (i, j) not in visited:
                a, b = self.bfs(heightMap, i, j, m, n)
                # print i, j, len(a)
                
                if b != float("Inf"):
                    visited.update(a)
                    for p, q in a:
                        total_sum += b-heightMap[p][q]
        
        return total_sum
    
    
