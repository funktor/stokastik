import math
class Solution(object):
    def numDupDigitsAtMostN(self, N):
        if N <= 10:
            return 0
        
        num_digits = int(math.log10(N))+1
        count, a, b = 0, 9, 9
        for i in range(2, num_digits):
            a = a*10
            b = b*(10-i+1)
            count += a-b
        
        num_str = str(N)
        a, b = 1, 1
        for i in reversed(range(len(num_str))):
            p = num_str[:i]
            q = set([int(x) for x in p])
            n = int(num_str[i])
            
            if len(p) == len(q):
                if i == 0:
                    r = set(range(1, n)).difference(q)
                    count += (n-1)*a-len(r)*b
                elif i == len(num_str)-1:
                    r = set(range(n+1)).difference(q)
                    count += (n+1)*a-len(r)*b
                else:
                    r = set(range(n)).difference(q)
                    count += n*a-len(r)*b
            else:
                if i == 0:
                    count += (n-1)*a
                elif i == len(num_str)-1:
                    count += (n+1)*a
                else:
                    count += n*a
                
            a = a*10
            b = b*(10-len(p))
        
        return count
