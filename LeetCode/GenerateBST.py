# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def generate(self, start, end, cached):
        if start == end:
            root = TreeNode(start)
            return [root]
        
        if start > end:
            return [None]
        
        trees = []
        
        for i in range(start, end+1):
            left_trees = self.generate(start, i-1, cached) if (start, i-1) not in cached else cached[(start, i-1)]
            right_trees = self.generate(i+1, end, cached) if (i+1, end) not in cached else cached[(i+1, end)]
            
            for x in left_trees:
                for y in right_trees:
                    curr_root = TreeNode(i)
                    curr_root.left = x
                    curr_root.right = y
                    trees.append(curr_root)
                    
        cached[(start, end)] = trees
        return trees
    
    def generateTrees(self, n):
        if n == 0:
            return []
        return self.generate(1, n, {})
        
