class Solution(object):

    def search_all_same(self, nums, left, right, target):
        x, y = True, True
        p, q = left, right

        while left <= right:
            mid = (left + right) / 2

            if nums[mid] == target:
                left = mid + 1
            else:
                x = False
                break

        left, right = p, q

        while left <= right:
            mid = (left + right) / 2

            if nums[mid] == target:
                if right == left + 1 and nums[right] != target:
                    y = False
                    break

                right = mid - 1
            else:
                y = False
                break

        return x and y


    def findMin(self, nums):
        if len(nums) == 1:
            return nums[0]

        possibility = float("Inf")

        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) / 2

            if nums[mid] <= nums[right]:
                if nums[mid] == nums[right] and self.search_all_same(nums, mid, right, nums[mid]) is False:
                    left = mid + 1
                else:
                    possibility = min(possibility, nums[mid])
                    right = mid - 1
            else:
                left = mid + 1

        return min(nums[left], possibility)

sol = Solution()
print sol.findMin([10,10,10,10,10,1,10,10,10])
