import math


class Solution(object):
    def reachNumber(self, target):
        if target == 0:
            return 0

        n = math.ceil((math.sqrt(8 * abs(target) - 1) - 1) / 2.0)
        m = int(n * (n + 1) / 2.0)

        if target % 2 == m % 2:
            return int(n)
        else:
            i = 1
            while target % 2 != m % 2:
                m += n + i
                i += 1

            return int(n + i - 1)