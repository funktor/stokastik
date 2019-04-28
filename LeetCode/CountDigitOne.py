import math

class Solution(object):
    def countDigitOne(self, n):
        if n < 1:
            return 0
        
#         m, sums = int(math.log10(n)), dict()
        
#         for m1 in range(1, m+1):
#             if m1 == 1:
#                 sums[m1] = 1
#             else:
#                 sums[m1] = 10*sums[m1-1] + 10**(m1-1)
                
        m1, out, u, v, p = 1, 0, 0, 1, 1
        while n > 0:
            rem = n % 10
            
            if m1 == 1:
                out += 1 if rem > 0 else 0
            else:
                if rem > 0:
                    if rem > 1:
                        out += rem*v + p
                    else:
                        out += v + 1 + u
                v = 10*v + p
                        
            u = rem*p + u
            n = int(n/10)
            p = 10*p
            m1 += 1
            
        return out
