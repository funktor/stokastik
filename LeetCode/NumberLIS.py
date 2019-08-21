class Solution(object):
    def findNumberOfLIS(self, nums):
        if len(nums) == 0:
            return 0
        
        seq_length = [(1, 1)]*len(nums)
        
        len_cnts_0, max_len_0 = {}, 1
        
        for i in range(len(nums)):
            len_cnts, max_len = {1:1}, 1
            if i > 0:
                for j in range(i):
                    if nums[i] > nums[j]:
                        a = seq_length[j][0] + 1
                        b = seq_length[j][1]
                        
                        if a not in len_cnts:
                            len_cnts[a] = 0
                        len_cnts[a] += b
                        max_len = max(max_len, a)

                seq_length[i] = (max_len, len_cnts[max_len])

            if max_len not in len_cnts_0:
                len_cnts_0[max_len] = 0

            len_cnts_0[max_len] += len_cnts[max_len]
            max_len_0 = max(max_len_0, max_len)
        
        return len_cnts_0[max_len_0]
