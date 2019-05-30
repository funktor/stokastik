class Solution(object):
    def minPatches(self, nums, n):
        curr_sum, i, num_patched = 0, 0, 0
        while curr_sum < n:
            if i >= len(nums) or nums[i] > curr_sum + 1:
                num_patched += 1
                curr_sum += curr_sum + 1
            else:
                curr_sum += nums[i]
                i += 1
                
        return num_patched
