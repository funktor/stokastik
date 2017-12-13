import collections

class Solution(object):
    def minCut(self, s):
        if len(s) == 0:
            return 0

        cache = collections.defaultdict(set)

        for length in range(1, len(s) + 1):
            for start in range(len(s) - length + 1):
                end = start + length - 1

                if length == 1:
                    cache[start].add(end)
                elif length == 2:
                    if s[start] == s[end]:
                        cache[start].add(end)
                else:
                    if s[start] == s[end] and end - 1 in cache[start + 1]:
                        cache[start].add(end)

        queue = [(0, 0)]
        visited = set()

        visited.add(0)

        while len(queue) > 0:
            q = queue.pop()
            start = q[0]

            if len(s) - 1 in cache[start]:
                return q[1]

            for end in cache[start]:
                if end + 1 not in visited:
                    queue.insert(0, (end + 1, q[1] + 1))
                    visited.add(end + 1)

        return 0

sol = Solution()
print sol.minCut("abakayakcat")