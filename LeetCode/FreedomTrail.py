import heapq

class Solution(object):
    def findRotateSteps(self, ring, key):
        heap = [(0, ring, key)]
        cache = {(ring, key):0}
        
        min_steps = float("Inf")
        
        while len(heap) > 0:
            z, x, y = heapq.heappop(heap)
            
            if len(y) == 0:
                min_steps = min(min_steps, z)
            
            else:
                if x[0] == y[0]:
                    if (x, y[1:]) not in cache or z+1 <= cache[(x, y[1:])]:
                        heapq.heappush(heap, (z+1, x, y[1:]))
                        cache[(x, y[1:])] = z+1
                else:
                    for i in range(len(x)):
                        if x[i] == y[0]:
                            a = x[i:]+x[:i]
                            if (a, y[1:]) not in cache or z+i+1 < cache[(a, y[1:])]:
                                heapq.heappush(heap, (z+i+1, a, y[1:]))
                                cache[(a, y[1:])] = z+i+1
                            break

                    for i in reversed(range(len(x))):
                        if x[i] == y[0]:
                            a = x[i:]+x[:i]
                            if (a, y[1:]) not in cache or z+len(x)-i+1 < cache[(a, y[1:])]:
                                heapq.heappush(heap, (z+len(x)-i+1, a, y[1:]))
                                cache[(a, y[1:])] = z+len(x)-i+1
                            break
        return min_steps
