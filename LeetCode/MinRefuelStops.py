import collections

class Solution(object):
    def minRefuelStops(self, target, startFuel, stations):
        stations = stations + [(target, 0)]
        n = len(stations)
        
        max_pos, max_step = [-1]*n, -1
        
        for i in range(n):
            result, tmp_max_pos = n+1, max_pos[:]
            steps = [step for step in range(n) if tmp_max_pos[step] != -1]
            
            if stations[i][0] <= startFuel:
                max_pos[0] = max(max_pos[0], startFuel+stations[i][1])
                result = 0
            
            for step in reversed(steps):
                if tmp_max_pos[step] >= stations[i][0]:
                    max_dist = tmp_max_pos[step]
                    result = min(result, step+1)
                    max_pos[step+1] = max(max_pos[step+1], max_dist+stations[i][1])
                    
                else:
                    break
        
        return -1 if result == n+1 else result
