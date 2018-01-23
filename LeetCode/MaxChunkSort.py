class Solution(object):
    def maxChunksToSorted(self, arr):
        max_cache, min_cache = [], []

        max_cache.append(arr[0])
        min_cache.append(arr[len(arr) - 1])

        for idx in range(1, len(arr)):
            a = max(arr[idx], max_cache[len(max_cache) - 1])
            max_cache.append(a)

        c = 1
        for idx in reversed(range(len(arr) - 1)):
            a = min(arr[idx], min_cache[c - 1])
            min_cache.append(a)
            c += 1

        min_cache = min_cache[::-1]

        count = 1

        for idx in range(len(arr) - 1):
            x, y = max_cache[idx], min_cache[idx + 1]
            if x <= y:
                count += 1

        return count


sol = Solution()
print sol.maxChunksToSorted([4])

