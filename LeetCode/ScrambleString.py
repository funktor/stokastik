import collections
class Solution(object):
    def is_anagram(self, s1, s2):
        if s1 == s2 or s1 == s2[::-1]:
            return True
        
        char_cnt1, char_cnt2 = collections.defaultdict(int), collections.defaultdict(int)
        
        for x in list(s1):
            char_cnt1[x] +=1
        for x in list(s2):
            char_cnt2[x] +=1
            
        if len(char_cnt1) != len(char_cnt2):
            return False
        
        for x, cnt in char_cnt1.items():
            if x not in char_cnt2 or char_cnt2[x] != cnt:
                return False
        
        return True
    
    def get_is_scrabled(self, s1, s2, cached):
        cached[(s1, s2)] = False
        
        if self.is_anagram(s1, s2) is False:
            cached[(s1, s2)] = False
        
        elif s1 == s2 or s1 == s2[::-1]:
            cached[(s1, s2)] = True
        
        else:
            for pos in range(1, len(s1)):
                a1, b1 = s1[:pos], s1[pos:]
                a2, b2 = s2[:pos], s2[pos:]

                a3, b3 = s1[:pos], s1[pos:]
                a4, b4 = s2[::-1][:pos], s2[::-1][pos:]
                
                if self.is_anagram(a1, a2):
                    if (a1, a2) in cached:
                        x = cached[(a1, a2)]
                    else:
                        x = self.get_is_scrabled(a1, a2, cached)
                        
                    if (b1, b2) in cached:
                        y = cached[(b1, b2)]
                    else:
                        y = self.get_is_scrabled(b1, b2, cached)
                        
                    z = x and y

                elif self.is_anagram(a3, a4):
                    if (a3, a4) in cached:
                        x = cached[(a3, a4)]
                    else:
                        x = self.get_is_scrabled(a3, a4, cached)
                        
                    if (b3, b4) in cached:
                        y = cached[(b3, b4)]
                    else:
                        y = self.get_is_scrabled(b3, b4, cached)
                        
                    z = x and y

                else:
                    z = False
                
                if z:
                    cached[(s1, s2)] = True
                    break
            
        return cached[(s1, s2)]
    
    def isScramble(self, s1, s2):
        cached = dict()
        return self.get_is_scrabled(s1, s2, cached)
