import collections

class Solution(object):
    def splitArray(self, nums, m):
        cache = collections.defaultdict(dict)

        prefix_sums = []

        for pos in range(len(nums)):
            if pos == 0:
                prefix_sums.append(nums[pos])
            else:
                prefix_sums.append(prefix_sums[pos - 1] + nums[pos])

        for i in range(1, m + 1):
            for pos in range(i - 1, len(nums)):
                if i == 1:
                    cache[pos][i] = prefix_sums[pos]
                else:
                    out = max(cache[pos - 1][i - 1], nums[pos])

                    for k in range(i - 2, pos - 1):
                        out = min(out, max(cache[k][i - 1], prefix_sums[pos] - prefix_sums[k]))

                    cache[pos][i] = out

        return cache[len(nums) - 1][m]