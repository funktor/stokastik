# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def deleteNode(self, root, key):
        if root is None:
            return None
        
        elif root.val == key:
            if root.left is None and root.right is None:
                return None
            elif root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                temp, parent, flag = root.right, root, False
                while temp.left is not None:
                    parent = temp
                    temp = temp.left
                    flag = True
                
                root.val = temp.val
                if flag:
                    parent.left = temp.right
                else:
                    parent.right = temp.right
                    
                return root
        else:
            if key < root.val:
                root.left = self.deleteNode(root.left, key)
            else:
                root.right = self.deleteNode(root.right, key)
                
            return root
        
