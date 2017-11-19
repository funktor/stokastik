import collections

class Solution(object):
    def checkSubarraySum(self, nums, k):
        remainder_dict = collections.defaultdict(int)

        last_rem = 0
        remainder_dict[0] = -1

        for idx in range(len(nums)):
            num = nums[idx]

            if k == 0:
                if last_rem + num == 0:
                    a = 0
                else:
                    a = last_rem - 1
            else:
                a = (last_rem + num) % k

            if a in remainder_dict and idx - remainder_dict[a] >= 2:
                return True

            last_rem = a

            if last_rem not in remainder_dict:
                remainder_dict[last_rem] = idx

        return False

sol = Solution()
print sol.checkSubarraySum([23, 2, 4, 6, 7], 42)