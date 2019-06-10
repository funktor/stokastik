# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def recoverFromPreorder(self, S):
        root_str, i = "", 0
        while i < len(S) and S[i] != "-":
            root_str += S[i]
            i += 1
                
        root = TreeNode(root_str)
        
        parent = dict()
        parent[root] = None
        
        curr_num_dashes, last_num_dashes = 0, 0
        
        while i < len(S):
            if S[i] != "-":
                
                curr_str = ""
                while i < len(S) and S[i] != "-":
                    curr_str += S[i]
                    i += 1
                    
                i -= 1
                
                if curr_num_dashes > last_num_dashes:
                    node = TreeNode(curr_str)
                    parent[node] = root
                    root.left = node
                    root = root.left

                else:
                    while last_num_dashes >= curr_num_dashes:
                        root = parent[root]
                        last_num_dashes -= 1

                    node = TreeNode(curr_str)
                    parent[node] = root
                    root.right = node
                    root = root.right

                last_num_dashes = curr_num_dashes
                curr_num_dashes = 0
                
            else:
                curr_num_dashes += 1
            
            i += 1
                
        while parent[root] != None:
            root = parent[root]
        
        return root
