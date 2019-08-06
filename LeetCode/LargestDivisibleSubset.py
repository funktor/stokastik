class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums) == 0:
            return []
        
        nums = sorted(nums)
        out = []
        for i in range(len(nums)):
            out.append([])
        
        for i in reversed(range(len(nums))):
            if i == len(nums)-1:
                out[i] = [nums[i]]
            else:
                max_len, max_len_out = -1, []
                for j in range(i+1, len(nums)):
                    x = out[j]
                    if x[0] % nums[i] == 0:
                        if len(x) > max_len:
                            max_len = len(x)
                            max_len_out = x
                
                out[i] = [nums[i]] + max_len_out
        
        output, max_len = [], -1
        for x in out:
            if len(x) > max_len:
                max_len = len(x)
                output = x
                
        return output
