class Solution(object):
    def distinctSubseqII(self, S):
        last_same_pos, pos = {}, {}
        m = 10**9 + 7
        
        for i in range(len(S)):
            if S[i] in pos:
                last_same_pos[i] = pos[S[i]]
            else:
                last_same_pos[i] = -1
            pos[S[i]] = i
        
        cnts = [0]*len(S)
        for i in range(len(S)):
            if i == 0:
                cnts[i] = 1
            else:
                cnt = 1 if last_same_pos[i] == -1 else 0
                start = last_same_pos[i] if last_same_pos[i] != -1 else 0
                
                for j in range(start, i):
                    cnt += cnts[j]%m
                
                cnts[i] = cnt%m
        
        sums = 0
        for cnt in cnts:
            sums += cnt%m
        
        return sums%m
        
                    
                
                
