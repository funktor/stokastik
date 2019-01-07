import heapq
def minRefuelStops(target, startFuel, stations):
    queue, max_dist, step, index = [], startFuel, 0, 0
    
    while True:
        while index < len(stations) and stations[index][0] <= max_dist:
            heapq.heappush(queue, -stations[index][1])
            index += 1
        if max_dist >= target:
            return step
        
        if len(queue) == 0:
            return -1
        
        max_dist += -heapq.heappop(queue)
        step += 1