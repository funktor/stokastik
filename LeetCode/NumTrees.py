class Solution(object):
    def generate(self, start, end, cached):
        if start >= end:
            return 1
        
        num_trees = 0
        
        for i in range(start, end+1):
            l = self.generate(start, i-1, cached) if (start, i-1) not in cached else cached[(start, i-1)]
            r = self.generate(i+1, end, cached) if (i+1, end) not in cached else cached[(i+1, end)]
            
            num_trees += l*r
                    
        cached[(start, end)] = num_trees
        return num_trees
    
    def numTrees(self, n):
        if n == 0:
            return 0
        return self.generate(1, n, {})
        
