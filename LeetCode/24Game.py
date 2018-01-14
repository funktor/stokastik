import collections

class Solution(object):

    def gcd(self, a, b):
        if a == 0:
            return b
        return self.gcd(b % a, a)

    def get_reduced(self, p, q):
        d = self.gcd(p, q)
        p, q = p / d, q / d

        return p, q

    def judgePoint24(self, nums):
        cache = collections.defaultdict(list)

        for length in range(1, 5):
            if length == 1:
                for idx in range(len(nums)):
                    num = nums[idx]
                    cache[length].append((num, 1, [idx]))
            else:
                for k in range(1, int(length / 2) + 1):
                    a, b = cache[k], cache[length - k]

                    for x in a:
                        for y in b:
                            k1, k2 = x[2], y[2]
                            h = set(k1).intersection(set(k2))

                            if len(h) == 0:
                                p, q = x[0] * y[1] + x[1] * y[0], x[1] * y[1]
                                out = self.get_reduced(p, q)

                                if length == 4:
                                    if out[0] == 24 and out[1] == 1:
                                        return True
                                else:
                                    cache[length].append((out[0], out[1], k1 + k2))

                                p, q = x[0] * y[1] - x[1] * y[0], x[1] * y[1]
                                out = self.get_reduced(p, q)

                                if length == 4:
                                    if out[0] == 24 and out[1] == 1:
                                        return True
                                else:
                                    cache[length].append((out[0], out[1], k1 + k2))

                                p, q = -x[0] * y[1] + x[1] * y[0], x[1] * y[1]
                                out = self.get_reduced(p, q)

                                if length == 4:
                                    if out[0] == 24 and out[1] == 1:
                                        return True
                                else:
                                    cache[length].append((out[0], out[1], k1 + k2))

                                p, q = x[0] * y[0], x[1] * y[1]
                                out = self.get_reduced(p, q)

                                if length == 4:
                                    if out[0] == 24 and out[1] == 1:
                                        return True
                                else:
                                    cache[length].append((out[0], out[1], k1 + k2))

                                if y[0] != 0:
                                    p, q = x[0] * y[1], x[1] * y[0]
                                    out = self.get_reduced(p, q)

                                    if length == 4:
                                        if out[0] == 24 and out[1] == 1:
                                            return True
                                    else:
                                        cache[length].append((out[0], out[1], k1 + k2))

                                if x[0] != 0:
                                    p, q = x[1] * y[0], x[0] * y[1]
                                    out = self.get_reduced(p, q)

                                    if length == 4:
                                        if out[0] == 24 and out[1] == 1:
                                            return True
                                    else:
                                        cache[length].append((out[0], out[1], k1 + k2))

        return False

sol = Solution()
print sol.judgePoint24([1,6,9,1])

