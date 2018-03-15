import collections

class Solution(object):
    def calculate_min_health(self, dungeon, start, cache):
        if start == (len(dungeon) - 1, len(dungeon[0]) - 1):
            cache[start] = dungeon[start[0]][start[1]] if dungeon[start[0]][start[1]] < 0 else 0

        else:
            p, q = (start[0] + 1, start[1]), (start[0], start[1] + 1)

            if p[0] < len(dungeon):
                if p in cache:
                    a = cache[p]
                else:
                    a = self.calculate_min_health(dungeon, p, cache)
            else:
                a = -float("Inf")

            if q[1] < len(dungeon[0]):
                if q in cache:
                    b = cache[q]
                else:
                    b = self.calculate_min_health(dungeon, q, cache)
            else:
                b = -float("Inf")

            c, d = dungeon[start[0]][start[1]] + a, dungeon[start[0]][start[1]] + b

            cache[start] = max(c, d) if c < 0 and d < 0 else 0

        return cache[start]

    def calculateMinimumHP(self, dungeon):
        cache = collections.defaultdict(dict)
        out = self.calculate_min_health(dungeon, (0,0), cache)

        return 1 - out

sol = Solution()
print sol.calculateMinimumHP([[-2,-3,3],[-5,-10,1],[10,30,-5]])
