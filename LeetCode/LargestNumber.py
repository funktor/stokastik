class Solution:

    def custom_merge(self, nums1, nums2):
        i, j = 0, 0
        out = []

        while i < len(nums1) and j < len(nums2):
            if int(str(nums1[i]) + str(nums2[j])) > int(str(nums2[j]) + str(nums1[i])):
                out.append(nums1[i])
                i += 1
            else:
                out.append(nums2[j])
                j += 1

        if i < len(nums1):
            for k in range(i, len(nums1)):
                out.append(nums1[k])

        if j < len(nums2):
            for k in range(j, len(nums2)):
                out.append(nums2[k])

        return out


    def custom_merge_sort(self, nums, left, right):
        if left >= right:
            return [nums[left]]
        else:
            mid = (left + right) / 2
            a = self.custom_merge_sort(nums, left, mid)
            b = self.custom_merge_sort(nums, mid + 1, right)

            return self.custom_merge(a, b)


    def largestNumber(self, nums):
        sorted_nums = self.custom_merge_sort(nums, 0, len(nums) - 1)

        if sorted_nums[0] == 0:
            return "0"

        out = ""
        for num in sorted_nums:
            out += str(num)

        return out

sol = Solution()
print sol.largestNumber([0,0])