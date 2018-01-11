class Solution(object):
    def findMin(self, nums):
        if len(nums) == 1:
            return nums[0]

        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) / 2

            if 1 <= mid <= len(nums) - 2:
                if nums[mid] < nums[mid - 1] and nums[mid] < nums[mid + 1]:
                    return nums[mid]
                elif nums[mid] < nums[right]:
                    right = mid - 1
                else:
                    left = mid + 1
            elif mid == 0:
                if nums[mid] < nums[mid + 1]:
                    return nums[mid]
                else:
                    left = mid + 1
            else:
                if nums[mid] < nums[mid - 1]:
                    return nums[mid]
                else:
                    right = mid - 1

        return nums[left]

sol = Solution()
print sol.findMin([3])
