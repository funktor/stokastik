import collections

class Solution(object):
    def isSelfCrossing(self, x):
        if len(x) < 4:
            return False
        
        for i in range(3, len(x)):
            a = x[i] >= x[i-2] and x[i-1] <= x[i-3]
            b = i >= 4 and x[i] + x[i-4] >= x[i-2] and x[i-2] > x[i-4] and x[i-1] == x[i-3]
            c = i >= 5 and x[i] + x[i-4] >= x[i-2] and x[i-2] > x[i-4] and x[i-1] + x[i-5] >= x[i-3] and x[i-3] > x[i-5] and x[i-1] <= x[i-3]
            
            if a or b or c:
                return True
            
        return False
