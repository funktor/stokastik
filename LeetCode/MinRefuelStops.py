class Solution(object):
    def minRefuelStops(self, target, startFuel, stations):
        n = len(stations)
        max_dist = [-1]*n

        for i in range(n):
            for step in reversed(range(i+1)):
                w = startFuel if step == 0 else max_dist[step-1]

                if w >= stations[i][0]:
                    max_dist[step] = max(max_dist[step], w + stations[i][1])
                else:
                    break

        for i in range(n):
            if max_dist[i] >= target:
                return i+1

        return -1