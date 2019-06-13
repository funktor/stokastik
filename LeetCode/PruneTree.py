# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def prune(self, root):
        if root is None:
            return 0
        
        sum_nodes_left = self.prune(root.left)
        sum_nodes_right = self.prune(root.right)
        
        if sum_nodes_left == 0:
            root.left = None
        if sum_nodes_right == 0:
            root.right = None
        
        if sum_nodes_left == 0 and sum_nodes_right == 0 and root.val == 0:
            root = None
            return 0
        
        return root.val + sum_nodes_left + sum_nodes_right
    
    def pruneTree(self, root):
        self.prune(root)
        return root
