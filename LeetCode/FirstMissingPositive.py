class Solution(object):

    def firstMissingPositive(self, nums):
        if len(nums) == 0:
            return 1

        for idx in range(len(nums)):
            x = int(nums[idx])

            if 1 <= x <= len(nums):
                nums[x - 1] = nums[x - 1] + 0.1

        for idx in range(len(nums)):
            if nums[idx] == int(nums[idx]):
                return idx + 1

        return len(nums) + 1


sol = Solution()
print sol.firstMissingPositive([3,4,-1,1])