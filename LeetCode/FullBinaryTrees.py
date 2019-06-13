# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def fbt(self, N, cache):
        if N % 2 == 0:
            cache[N] = []
            
        elif N == 1:
            cache[N] = [TreeNode(0)]
        
        else:
            trees = []

            for i in range(1, N-1):
                if i not in cache:
                    left_fbt = self.fbt(i, cache)
                else:
                    left_fbt = cache[i]

                if N-i-1 not in cache:
                    right_fbt = self.fbt(N-i-1, cache)
                else:
                    right_fbt = cache[N-i-1]

                for left_root in left_fbt:
                    for right_root in right_fbt:
                        root = TreeNode(0)
                        root.left = left_root
                        root.right = right_root
                        trees.append(root)
        
            cache[N] = trees
            
        return cache[N]
    
    def allPossibleFBT(self, N):
        return self.fbt(N, {})
