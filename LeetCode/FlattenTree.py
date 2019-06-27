# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flat(self, root):
        if root is None:
            return None, None
        
        root.left, last_left = self.flat(root.left)
        root.right, last_right = self.flat(root.right)
        
        if last_left is not None:
            last_left.right = root.right
            root.right = root.left
            root.left = None
            
            temp = last_right if last_right is not None else last_left

            return root, temp
        elif last_right is not None:
            return root, last_right
        else:
            return root, root
        
    def flatten(self, root):
        return self.flat(root)[0]
        
