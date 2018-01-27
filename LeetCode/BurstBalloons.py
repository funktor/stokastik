import collections

class Solution(object):
    def maxCoins(self, nums):
        if len(nums) == 0:
            return 0
        
        if len(nums) == 1:
            return nums[0]

        cache = collections.defaultdict(dict)

        for length in range(1, len(nums) + 1):
            for start in range(len(nums) - length + 1):
                end = start + length - 1

                if length == 1:
                    if start == 0:
                        cache[start][end] = nums[start] * nums[start + 1]
                    elif start == len(nums) - 1:
                        cache[start][end] = nums[start] * nums[start - 1]
                    else:
                        cache[start][end] = nums[start - 1] * nums[start] * nums[start + 1]

                else:
                    max_coins = -float("Inf")

                    for pos in range(start, end + 1):

                        if start - 1 >= 0 and end + 1 <= len(nums) - 1:
                            x = nums[start - 1] * nums[pos] * nums[end + 1]
                        elif start - 1 >= 0:
                            x = nums[pos] * nums[start - 1]
                        elif end + 1 <= len(nums) - 1:
                            x = nums[pos] * nums[end + 1]
                        else:
                            x = nums[pos]

                        if pos == start:
                            u = cache[pos + 1][end]
                            max_coins = max(max_coins, u + x)

                        elif pos == end:
                            u = cache[start][pos - 1]
                            max_coins = max(max_coins, u + x)

                        else:
                            u, v = cache[start][pos - 1], cache[pos + 1][end]
                            max_coins = max(max_coins, u + v + x)

                    cache[start][end] = max_coins

        return cache[0][len(nums) - 1]



sol = Solution()
print sol.maxCoins([5,7])
