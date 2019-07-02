import collections

class Solution(object):
    def strangePrinter(self, s):
        if len(s) == 0:
            return 0
        
        cache = collections.defaultdict(dict)

        for length in range(1, len(s)+1):
            for i in range(len(s)-length+1):
                j = i+length-1
                if length == 1:
                    cache[i][j] = 1
                else:
                    if s[j-1] == s[j]:
                        cache[i][j] = cache[i][j-1]
                    else:
                        min_turns = cache[i][j-1]+1
                        for k in range(i, j):
                            if s[k] == s[j]:
                                if k == i:
                                    min_turns = min(min_turns, cache[k+1][j])
                                else:
                                    a = cache[i][k-1]
                                    b = cache[k][j]
                                    min_turns = min(min_turns, a + b)
                        
                        cache[i][j] = min_turns
        
        return cache[0][len(s)-1]
                                
