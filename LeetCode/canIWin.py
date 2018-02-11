import collections

class Solution(object):
    def can_win(self, nums, n, cached):
        if len(nums) > 0 and n <= nums[-1]:
            return True

        elif tuple(nums) in cached:
            return cached[tuple(nums)]

        else:
            sum = 0
            for num in nums:
                sum += num

            if sum == n:
                out = True if len(nums) % 2 == 1 else False
                cached[tuple(nums)] = out

                return out

            flag = False

            for idx in range(len(nums)):
                new_nums = nums[:idx] + nums[idx + 1:]
                rep = tuple(new_nums)

                m = n - nums[idx]

                if rep in cached:
                    out = cached[rep]

                else:
                    out = self.can_win(new_nums, m, cached)
                    cached[rep] = out

                if out is False:
                    flag = True
                    break


            cached[tuple(nums)] = flag

            return flag


    def canIWin(self, maxChoosableInteger, desiredTotal):
        if desiredTotal == 0:
            return True

        if maxChoosableInteger * (maxChoosableInteger + 1) / 2 < desiredTotal:
            return False

        cached = collections.defaultdict(int)
        nums = range(1, maxChoosableInteger + 1)

        return self.can_win(nums, desiredTotal, cached)


sol = Solution()
print sol.canIWin(20, 209)


