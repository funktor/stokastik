import collections

class Solution(object):

    def validTree(self, n, edges):
        if len(edges) == 0:
            if n == 1:
                return True
            else:
                return False

        if n == 1 and len(edges) == 1:
            return True

        edge_map = collections.defaultdict(set)

        for edge in edges:
            start, end = edge
            edge_map[start].add(end)
            edge_map[end].add(start)

        start = edges[0][0]

        queue = [start]
        visited_map = collections.defaultdict(set)
        visited_map[start] = set()

        while len(queue) > 0:
            q = queue.pop()

            children = edge_map[q]

            for child in children:

                if child in visited_map and child not in visited_map[q]:
                    return False

                if child not in visited_map:
                    queue.insert(0, child)
                    visited_map[child].add(q)

        if len(visited_map) == n:
            return True

        return False

sol = Solution()
print sol.validTree(5, [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]])