import math

class Solution(object):
    def swap_in_place(self, nums):
        if len(nums) > 2:
            n = len(nums)

            if n % 2 == 0:
                mid = int(n / 2)
            else:
                mid = int(n / 2) + 1

            nums1, nums2 = nums[:mid], nums[mid:]

            i, j = len(nums1) - 1, len(nums2) - 1
            c = 0
            while i >= 0 or j >= 0:
                if c % 2 == 0 and i >= 0:
                    nums[c] = nums1[i]
                    i -= 1
                else:
                    nums[c] = nums2[j]
                    j -= 1

                c += 1

    def wiggleSort(self, nums):
        if len(nums) > 1:
            nums.sort()
            self.swap_in_place(nums)