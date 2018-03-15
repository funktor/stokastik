import collections

class Solution(object):
    def kInversePairs(self, n, k):
        cache = collections.defaultdict(dict)
        modulo = 10**9 + 7

        for m in range(1, n + 1):
            u = int(m * (m - 1) / 2)
            for q in range(k + 1):
                if q == 0:
                    cache[m][q] = 1

                elif q > u:
                    cache[m][q] = 0

                else:
                    if q - m >= 0:
                        cache[m][q] = (cache[m][q - 1] + cache[m - 1][q] - cache[m - 1][q - m]) % modulo
                    else:
                        cache[m][q] = (cache[m][q - 1] + cache[m - 1][q]) % modulo

        return cache[n][k]

sol = Solution()
print sol.kInversePairs(1000, 1000)


