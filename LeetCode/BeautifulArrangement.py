import collections

class Solution(object):
    def countArrangement(self, N):
        h = {}
        for i in range(1, N+1):
            if i == 1:
                h[i] = set(range(1, N+1))
            else:
                h[i] = set()
                for j in range(1, i):
                    if i in h[j]:
                        h[i].add(j)
                for j in range(1, N/i+1):
                    h[i].add(i*j)
                    
        queue = collections.deque([(0, set())])
        visited, count = set([0]), 0
        
        while len(queue) > 0:
            curr_i, nums = queue.popleft()
            
            if curr_i == N:
                count += 1
            
            else:
                valid_nums = h[curr_i+1]
                for x in (valid_nums - nums):
                    new_nums = nums.copy()
                    new_nums.add(x)
                    queue.append((curr_i+1, new_nums))
                    
        return count
