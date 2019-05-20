class Solution(object):
    def has_dup(self, S, length, hashes):
        encounters = dict()
        d, q, h, g = 256, 32416190071, 1, 0
        
        for i in range(length): 
            h = (h*d)%q
        
        for i in range(len(S)-length+1):
            g = (hashes[i+length-1]-hashes[i-1]*h)%q if i > 0 else hashes[i+length-1]%q
            if g < 0: 
                g += q
            if g not in encounters:
                encounters[g] = []
            encounters[g].append(i)
        
        out = None
        for g, pos_list in encounters.items():
            if len(pos_list) > 1:
                v = set()
                for i in pos_list:
                    if S[i:i+length] in v:
                        return S[i:i+length]
                    v.add(S[i:i+length])
        return None
    
                        
    def longestDupSubstring(self, S):
        if len(set(S)) == 1:
            return S[:len(S)-1]
        
        d, q = 256, 32416190071
        encounters = dict()
        
        hashes = [0]*len(S)
        for i in range(len(S)):
            if i == 0:
                hashes[0] = ord(S[0])%q
            else:
                hashes[i] = (d*hashes[i-1] + ord(S[i]))%q
        
        left, right = 1, len(S)-1
        res = ""
        
        while left <= right:
            mid = (left + right)/2
            out = self.has_dup(S, mid, hashes)
            if out is None:
                right = mid-1
            else:
                res = out
                left = mid+1
        
        return res
