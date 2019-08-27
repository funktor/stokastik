class Solution(object):
    def decompose(self, text, start, end, cache):
        if start > end:
            return 0
        
        i, j = start, end
        prefix, suffix = "", ""
        
        max_cnt = 1
        
        while i < j:
            prefix += text[i]
            suffix = text[j] + suffix
            
            if prefix == suffix:
                x = self.decompose(text, i+1, j-1, cache) if (i+1, j-1) not in cache else cache[(i+1, j-1)]
                max_cnt = max(max_cnt, 2+x)
            
            i += 1
            j -= 1
        
        cache[(start, end)] = max_cnt
        return cache[(start, end)]
    
    def longestDecomposition(self, text):
        return self.decompose(text, 0, len(text)-1, {})
