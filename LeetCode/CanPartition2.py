class Solution(object):
    def canPartition(self, nums):
        sum_nums = 0
        for i in range(len(nums)):
            sum_nums += nums[i]
            
        if sum_nums % 2 == 1:
            return False
        
        m = sum_nums/2
        partition_sums = [[False]*len(nums) for i in range(m+1)]
        
        for sums in range(1, m+1):
            for i in range(len(nums)):
                if i == 0:
                    partition_sums[sums][i] = 1 if nums[i] == sums else 0
                else:
                    partition_sums[sums][i] = partition_sums[sums][i-1] or partition_sums[sums-nums[i]][i-1] or nums[i] == sums
        
        return partition_sums[m][len(nums)-1]
