import collections

class Solution(object):
    def coinChange(self, coins, amount):
        if amount == 0:
            return 0

        cache = collections.defaultdict()
        coins_set = set(coins)

        for amt in range(1, amount + 1):
            if amt in coins_set:
                cache[amt] = 1
            else:
                min_change = amount + 1

                for coin in coins:
                    if amt > coin:
                        min_change = min(min_change, 1 + cache[amt - coin])

                cache[amt] = min_change

        if cache[amount] == amount + 1:
            return -1

        return cache[amount]


sol = Solution()
print sol.coinChange([2], 4)

