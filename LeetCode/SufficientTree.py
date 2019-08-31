# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sufficient(self, root, limit):
        if root.left is None and root.right is None:
            return root.val, root.val
        
        elif root.left is None:
            rs1, rs2 = self.sufficient(root.right, limit-root.val)
            if rs1 < limit-root.val and rs2 < limit-root.val:
                root.right = None
            
            return -float("Inf"), root.val + max(rs1, rs2)
        
        elif root.right is None:
            ls1, ls2 = self.sufficient(root.left, limit-root.val)
            if ls1 < limit-root.val and ls2 < limit-root.val:
                root.left = None
            
            return root.val + max(ls1, ls2), -float("Inf")
        
        else:
            ls1, ls2 = self.sufficient(root.left, limit-root.val)
            rs1, rs2 = self.sufficient(root.right, limit-root.val)

            if ls1 < limit-root.val and ls2 < limit-root.val:
                root.left = None

            if rs1 < limit-root.val and rs2 < limit-root.val:
                root.right = None

            return root.val + max(ls1, ls2), root.val + max(rs1, rs2)
            
            
    def sufficientSubset(self, root, limit):
        if root is None:
            return None
        
        ls, rs = self.sufficient(root, limit)
        if ls < limit and rs < limit:
            return None
        return root
        
