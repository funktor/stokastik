class Solution(object):
    def max_rob(self, nums, i, cached):
        if i == len(nums)-1:
            return nums[i], nums[i]
        
        if i >= len(nums):
            return 0, 0
        
        a, x = self.max_rob(nums, i+1, cached) if i+1 not in cached else cached[i+1]
        b, y = self.max_rob(nums, i+2, cached) if i+2 not in cached else cached[i+2]
        
        c = nums[i] + y
        
        cached[i] = (c, max(a, c))
        return cached[i]
        
    def rob(self, nums):
        if len(nums) == 1:
            return nums[0]
        
        a, x = self.max_rob(nums[:len(nums)-1], 0, {})
        b, y = self.max_rob(nums[1:], 0, {})
        return max(x, y)
        
