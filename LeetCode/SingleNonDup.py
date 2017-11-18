class Solution(object):
    def singleNonDuplicate(self, nums):
        left, right = 0, len(nums)-1

        while left <= right:
            mid = (left + right) / 2

            if mid % 2 == 1 and mid - 1 >= 0 and nums[mid] == nums[mid - 1]:
                left = mid + 1
            elif mid % 2 == 0 and mid + 1 < len(nums) and nums[mid] == nums[mid + 1]:
                left = mid + 2
            elif mid % 2 == 1 and mid + 1 < len(nums) and nums[mid] == nums[mid + 1]:
                right = mid - 1
            else:
                right = mid - 2

        return nums[left]

sol = Solution()
print sol.singleNonDuplicate([1,1,2,2,3,3,4,4,5,5,6])