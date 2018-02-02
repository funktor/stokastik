import collections


class Solution(object):
    def maxProduct(self, nums):
        cache = collections.defaultdict(list)

        for start in reversed(range(len(nums))):
            if start == len(nums) - 1:
                a = nums[start] if nums[start] < 0 else 0
                b = nums[start] if nums[start] > 0 else 0
                c = nums[start]
                d = start

            else:
                x, y, z, w = cache[start + 1]

                if nums[start] == 0:
                    a, b = 0, 0
                    if z <= 0:
                        c, d = 0, start
                    else:
                        c, d = z, w

                else:
                    if nums[start] > 0:
                        a, b = nums[start] * x, max(nums[start], nums[start] * y)
                    else:
                        a, b = min(nums[start], nums[start] * y), max(nums[start], nums[start] * x)

                    if w == start + 1:
                        if b >= z and b >= nums[start] * z:
                            c = b
                            d = start

                        elif nums[start] * z >= z and nums[start] * z >= b:
                            c = nums[start] * z
                            d = start

                        else:
                            c = z
                            d = w

                    else:
                        if b >= z:
                            c = b
                            d = start
                        else:
                            c = z
                            d = w

            cache[start] = [a, b, c, d]

        return cache[0][2]