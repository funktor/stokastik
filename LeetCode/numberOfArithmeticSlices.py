import collections

class Solution(object):
    def numberOfArithmeticSlices(self, A):
        cache = collections.defaultdict(lambda: collections.defaultdict(lambda : 0))
        total = 0

        for pos in reversed(range(len(A) - 1)):
            for pos2 in range(pos + 1, len(A)):
                cache[pos][A[pos2] - A[pos]] += 1

                if A[pos2] - A[pos] in cache[pos2]:
                    cache[pos][A[pos2] - A[pos]] += cache[pos2][A[pos2] - A[pos]]
                    total += cache[pos2][A[pos2] - A[pos]]

        return total

sol = Solution()
print sol.numberOfArithmeticSlices([2,4,6,8,10])