import collections
class Solution(object):
    def minDistance(self, word1, word2):
        cached = collections.defaultdict(dict)
        
        for i in range(-1, len(word1)):
            for j in range(-1, len(word2)):
                if i == -1 and j == -1:
                    cached[i][j] = 0
                    
                elif i == -1:
                    cached[i][j] = j+1
                
                elif j == -1:
                    cached[i][j] = i+1
                    
                elif word1[i] == word2[j]:
                    cached[i][j] = cached[i-1][j-1]
                    
                else:
                    a = 1 + cached[i][j-1]
                    b = 1 + cached[i-1][j]
                    c = 1 + cached[i-1][j-1]
                    
                    cached[i][j] = min(a, b, c)
        
        return cached[len(word1)-1][len(word2)-1]
