# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import numpy as np


class Solution(object):
    def isBST(self, root):
        if root.left is None and root.right is None:
            return True, root.val, root.val

        elif root.left is None:
            a, min_val, max_val = self.isBST(root.right)
            return a and root.val < min_val, min(root.val, min_val), max(root.val, max_val)

        elif root.right is None:
            a, min_val, max_val = self.isBST(root.left)
            return a and root.val > max_val, min(root.val, min_val), max(root.val, max_val)

        else:
            a, min_val_l, max_val_l = self.isBST(root.left)
            b, min_val_r, max_val_r = self.isBST(root.right)

            return a and b and max_val_l < root.val < min_val_r, min(root.val, min_val_l, min_val_r), max(root.val,
                                                                                                          max_val_l,
                                                                                                          max_val_r)

    def isValidBST(self, root):
        if root is None:
            return True

        out, x, y = self.isBST(root)

        return out
