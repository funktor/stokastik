class Solution(object):
    def findDuplicates(self, nums):
        if len(nums) == 0:
            return []

        num_set = set(nums)

        for idx in range(len(nums)):
            q = abs(nums[idx]) - 1
            nums[q] *= -1

        out = []
        for idx in range(len(nums)):
            if nums[idx] > 0 and (idx + 1) in num_set:
                out.append(idx + 1)

        return out

sol = Solution()
print sol.findDuplicates([5,4,6,7,9,3,10,9,5,6])