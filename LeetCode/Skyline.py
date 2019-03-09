import heapq
class Solution(object):
    def getSkyline(self, buildings):
        if len(buildings) == 0:
            return []
        
        heapq.heapify(buildings)
        key_points = []
        
        while len(buildings) > 0:
            x = heapq.heappop(buildings)
            if len(buildings) == 0:
                key_points.append(x)
                break
                
            y = buildings[0]
            
            if x[1] < y[0]:
                key_points.append(x)
                key_points.append([x[1], y[0], 0])
            
            else:
                if x[2] == y[2]:
                    y = heapq.heappop(buildings)
                    z = x
                    if x[1] < y[1]:
                        z = [x[0], y[1], x[2]]
                    heapq.heappush(buildings, z)
                        
                elif x[2] < y[2]:
                    if y[0] > x[0]:
                        key_points.append([x[0], y[0], x[2]])
                        
                    if x[1] > y[1]:
                        heapq.heappush(buildings, [y[1], x[1], x[2]])
                        
                else:
                    if x[1] == y[0]:
                        key_points.append(x)
                    else:
                        y = heapq.heappop(buildings)
                        if x[1] < y[1]:
                            y[0] = x[1]
                            heapq.heappush(buildings, y)
                        heapq.heappush(buildings, x)  
        
        last = key_points[-1]
        key_points = [[x, y] for x, _, y in key_points] + [[last[1], 0]]
        
        return key_points
