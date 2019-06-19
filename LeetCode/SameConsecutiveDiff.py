import collections

class Solution(object):
    def numsSameConsecDiff(self, N, K):
        cached = collections.defaultdict(dict)
        
        for length in range(1, N+1):
            for i in range(10):
                if length == 1:
                    cached[length][i] = [str(i)]
                else:
                    out = []
                    
                    if i+K <= 9:
                        for x in cached[length-1][i+K]:
                            out.append(str(i) + x)
                    
                    if K > 0 and i-K >= 0:
                        for x in cached[length-1][i-K]:
                            out.append(str(i) + x)
                            
                    cached[length][i] = out
        
        out = []
        for i, x in cached[N].items():
            if (N > 1 and i > 0) or N==1:
                for y in x:
                    out.append(int(y))
                
        return out
                
        
        
