import math

class Solution(object):
    def racecar(self, target):
        heap = [(target, 0, 1, 0)]
        cache = {(0, 1) : 0}
        
        min_dist = float("Inf")
        w = int(math.log(target, 2))
        a, b = 0, target+2**w-1
        
        while len(heap) > 0:
            cost, pos, speed, dist = heapq.heappop(heap)
            
            if pos == target:
                min_dist = min(min_dist, dist)
            
            new_pos, new_speed = pos + speed, 2*speed
            if ((new_pos, new_speed) not in cache or cache[(new_pos, new_speed)] > dist+1) and a <= new_pos <= b and dist+1 < min_dist:
                heapq.heappush(heap, (dist + abs(target-new_pos), new_pos, new_speed, dist+1))
                cache[(new_pos, new_speed)] = dist+1
            
            if speed > 0:
                new_pos, new_speed = pos, -1
            else:
                new_pos, new_speed = pos, 1
            
            if ((new_pos, new_speed) not in cache or cache[(new_pos, new_speed)] > dist+1) and a <= new_pos <= b and dist+1 < min_dist:
                heapq.heappush(heap, (dist + abs(target-new_pos), new_pos, new_speed, dist+1))
                cache[(new_pos, new_speed)] = dist+1
        
        return min_dist
