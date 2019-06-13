# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def bst(self, root, parent_val):
        if root is None:
            return 0
        
        sum_nodes_right = self.bst(root.right, parent_val)
        x = root.val
        root.val += parent_val + sum_nodes_right
        sum_nodes_left = self.bst(root.left, root.val)
        
        return sum_nodes_left + sum_nodes_right + x
        
    def bstToGst(self, root):
        self.bst(root, 0)
        return root
        
