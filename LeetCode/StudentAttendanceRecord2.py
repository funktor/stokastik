from collections import deque

class Solution(object):
    def checkRecord(self, n):
        p = 10**9 + 7
        
        num_rewardable, num_a, num_ll, num_l = deque([]), deque([]), 0, 0
        for m in range(1, n+1):
            if m == 1:
                num_rewardable.append(3)
                num_a.append(1)
            elif m == 2:
                x = num_rewardable[-1]
                num_rewardable.append(3*x - num_a[-1])
                num_a.append(num_a[-1] + x)
                num_ll, num_l = 1, 2
            elif m == 3:
                x = num_rewardable[-1]
                num_rewardable.append(3*x - num_a[-1] - num_ll)
                num_a.append(num_a[-1] + x)
                num_ll = num_l
                num_l = 2*num_rewardable[-3]-num_a[-3]
            elif m == 4:
                x = num_rewardable[-1]
                num_rewardable.append(3*x - num_a[-1] - num_ll)
                num_a.append(num_a[-1] + x - 1)
                num_ll = num_l
                num_l = 2*num_rewardable[-3]-num_a[-3]
            else:
                x = num_rewardable[-1]
                num_rewardable.append((3*x - num_a[-1] - num_ll)%p)
                num_a.append((num_a[-1] + x - num_rewardable[-5])%p)
                num_ll = num_l
                num_l = (2*num_rewardable[-3]-num_a[-3])%p
            
            if len(num_a) > 2:
                num_a.popleft()
            
            if len(num_rewardable) > 4:
                num_rewardable.popleft()
                
        return num_rewardable[-1]%p
