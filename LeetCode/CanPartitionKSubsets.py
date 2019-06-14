class Solution(object):
    def dfs(self, g, h, index, n, visited):
        visited.add(index)
        
        if len(g) == n:
            return True
        
        flag = False
        for i in range(len(h)):
            if i not in visited and len(g.intersection(set(h[i]))) == 0:
                new_visited = visited.copy()
                g1 = g.copy()
                g1.update(h[i])
                flag = flag or self.dfs(g1, h, i, n, new_visited)
        
        return flag
    
    def canPartitionKSubsets(self, nums, k):
        sum_nums = 0
        for i in range(len(nums)):
            sum_nums += nums[i]
            
        if sum_nums % k == 1:
            return False
        
        m = sum_nums/k
        partition_sums = [[False]*len(nums) for i in range(m+1)]
        partitions = [[[]]*len(nums) for i in range(m+1)]
        
        for sums in range(1, m+1):
            for i in range(len(nums)):
                if i == 0:
                    partition_sums[sums][i] = nums[i] == sums
                    partitions[sums][i] = [[i]] if nums[i] == sums else []
                else:
                    partition_sums[sums][i] = partition_sums[sums][i-1] or (sums-nums[i] >= 1 and partition_sums[sums-nums[i]][i-1]) or nums[i] == sums
                    
                    if partition_sums[sums][i]:
                        if partitions[sums][i-1]:
                            a = partitions[sums][i-1]
                        else:
                            a = []
                            
                        if sums-nums[i] > 0:
                            b = []
                            for x in partitions[sums-nums[i]][i-1]:
                                if len(x) > 0:
                                    b.append(x + [i])
                        else:
                            b = []
                        
                        if nums[i] == sums:
                            c = [[i]]
                        else:
                            c = []
                        
                        partitions[sums][i] = a + b + c
                        
        
        if partition_sums[m][len(nums)-1] is False or len(partitions[m][len(nums)-1]) < k:
            return False
        
        h = partitions[m][len(nums)-1]
        
        flag = False
        for i in range(len(h)):
            flag = flag or self.dfs(set(h[i]), h, i, len(nums), set())
        return flag
