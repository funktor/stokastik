# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distribute_moves(self, root):
        if root is None:
            return 0, 0
        
        l, c1 = self.distribute_moves(root.left)
        r, c2 = self.distribute_moves(root.right)
        
        p = root.val+l+r-1
        c = c1 + c2 + abs(p)
        
        return p, c
        
    def distributeCoins(self, root):
        p, c = self.distribute_moves(root)
        return c
