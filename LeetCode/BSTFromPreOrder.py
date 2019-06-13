# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# [9,5,1,7,8,10,12]
class Solution(object):
    def bstFromPreorder(self, preorder):
        root = TreeNode(preorder[0])
        next_highest_parent, parent = dict(), dict()
        next_highest_parent[root] = None
        parent[root] = None
        
        for i in range(1, len(preorder)):
            if preorder[i] <= root.val:
                root.left = TreeNode(preorder[i])
                next_highest_parent[root.left] = root
                parent[root.left] = root
                root = root.left
                
            else:
                while root is not None and next_highest_parent[root] is not None and preorder[i] > next_highest_parent[root].val:
                    root = next_highest_parent[root]
                
                root.right = TreeNode(preorder[i])
                next_highest_parent[root.right] = next_highest_parent[root]
                parent[root.right] = root
                root = root.right
        
        while root is not None and parent[root] is not None:
            root = parent[root]
        
        return root
