class Solution(object):

    def find_optimum_point(self, sorted_nums, pair_val):

        left, right = 0, len(sorted_nums) - 1
        first_val = sorted_nums[0]

        while left < right:
            mid = (left + right) / 2

            if sorted_nums[mid] - first_val == pair_val:
                return mid

            elif sorted_nums[mid] - first_val < pair_val:
                left = mid + 1

            else:
                right = mid - 1

        if sorted_nums[left] - first_val > pair_val:
            return left - 1

        return left

    def num_pairs_smaller(self, sorted_nums, pair_val):

        optimum_pt = self.find_optimum_point(sorted_nums, pair_val)
        found_exact = False

        length = optimum_pt + 1

        if length % 2 == 0:
            cnt = (length / 2) * (length - 1)
        else:
            cnt = ((length - 1) / 2) * length

        if sorted_nums[optimum_pt] - sorted_nums[0] == pair_val:
            found_exact = True

        end_pt = optimum_pt + 1
        start_pt = 1

        while end_pt < len(sorted_nums):
            if sorted_nums[end_pt] - sorted_nums[start_pt] <= pair_val:
                if sorted_nums[end_pt] - sorted_nums[start_pt] == pair_val:
                    found_exact = True

                cnt += end_pt - start_pt
                end_pt += 1
            else:
                while start_pt < end_pt and sorted_nums[end_pt] - sorted_nums[start_pt] > pair_val:
                    start_pt += 1

        return cnt, found_exact

    def list_pairs(self, nums, k):

        pairs = []
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                pairs.append(abs(nums[i] - nums[j]))

        pairs = sorted(pairs)

        return pairs, pairs[k-1]


    def smallestDistancePair(self, nums, k):
        sorted_nums = sorted(nums)

        arr = range(sorted_nums[len(sorted_nums)-1] - sorted_nums[0] + 1)

        left, right = 0, len(arr) - 1

        while left <= right:
            mid = (left + right) / 2

            q, is_exact = self.num_pairs_smaller(sorted_nums, arr[mid])

            if mid < len(arr)-1 and q == k and self.num_pairs_smaller(sorted_nums, arr[mid + 1])[0] > k and is_exact:
                return arr[mid]

            elif mid < len(arr)-1 and q < k and self.num_pairs_smaller(sorted_nums, arr[mid + 1])[0] > k:
                return arr[mid + 1]

            elif q < k:
                left = mid + 1

            else:
                right = mid - 1

        return arr[left]



sol = Solution()
print sol.smallestDistancePair([9,10,7,10,6,1,5,4,9,8], 13)