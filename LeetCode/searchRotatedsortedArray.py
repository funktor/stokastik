class Solution(object):
    def search(self, nums, target):
        if len(nums) == 1:
            if nums[0] == target:
                return 0
            else:
                return -1

        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) / 2

            if nums[mid] == target:
                return mid

            elif nums[mid] <= nums[left] and nums[mid] <= nums[right]:
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

            elif nums[mid] >= nums[left] and nums[mid] >= nums[right]:
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1

            else:
                if target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1

        return -1

sol = Solution()
print sol.search([9,10,1,2,3,4,5,6,7,8], 11)