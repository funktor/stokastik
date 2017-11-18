class Solution(object):

    def search_less_than_equal(self, arr, num, left, right):

        while left <= right:
            mid = (left + right) / 2

            if arr[mid] <= num and ((mid + 1 <= right and arr[mid + 1] > num) or mid == right):
                return mid
            elif arr[mid] < num:
                left = mid + 1
            else:
                right = mid - 1

        return right

    def search_less_than(self, arr, num, left, right):

        while left <= right:
            mid = (left + right) / 2

            if arr[mid] < num and ((mid + 1 <= right and arr[mid + 1] >= num) or mid == right):
                return mid
            elif arr[mid] < num:
                left = mid + 1
            else:
                right = mid - 1

        return right


    def get_median(self, nums1, nums2, median_pos):

        left_1, left_2 = 0, 0
        right_1, right_2 = len(nums1) - 1, len(nums2) - 1

        while left_1 <= right_1 or left_2 <= right_2:

            if left_1 <= right_1:
                mid_1 = (left_1 + right_1) / 2

                idx2 = self.search_less_than(nums2, nums1[mid_1], left_2, right_2)

                w = idx2 + mid_1 + 1

                if w == median_pos:
                    return nums1[mid_1]

                elif w < median_pos:
                    left_1 = mid_1 + 1
                    left_2 = idx2 + 1
                else:
                    right_1 = mid_1 - 1
                    right_2 = idx2

            if left_2 <= right_2:
                mid_2 = (left_2 + right_2) / 2

                idx1 = self.search_less_than_equal(nums1, nums2[mid_2], left_1, right_1)

                w = idx1 + mid_2 + 1

                if w == median_pos:
                    return nums2[mid_2]

                elif w < median_pos:
                    left_2 = mid_2 + 1
                    left_1 = idx1 + 1
                else:
                    right_2 = mid_2 - 1
                    right_1 = idx1

        return -1


    def findMedianSortedArrays(self, nums1, nums2):

        n = len(nums1) + len(nums2)

        if n % 2 == 1:
            return self.get_median(nums1, nums2, n/2)
        else:
            a = self.get_median(nums1, nums2, n/2 - 1)
            b = self.get_median(nums1, nums2, n/2)

            if a == -1 and b == -1:
                return None

            return (a + b) / float(2.0)

sol = Solution()
print sol.findMedianSortedArrays([], [1,2,3])