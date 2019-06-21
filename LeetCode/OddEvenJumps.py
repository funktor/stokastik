class TreeNode(object):
    def __init__(self, val, index):
        self.val = val
        self.index = index
        self.left, self.right = None, None
        
class Solution(object):
    def insert(self, root, val, index):
        if root is None:
            return TreeNode(val, index), None, None
        
        else:
            if val == root.val:
                curr_left, curr_right = root.index, root.index
                root.index = index
                
            elif val < root.val:
                root.left, curr_left, curr_right = self.insert(root.left, val, index)
                curr_left = root.index if curr_left is None else curr_left
                
            else:
                root.right, curr_left, curr_right = self.insert(root.right, val, index)
                curr_right = root.index if curr_right is None else curr_right
            
            return root, curr_left, curr_right
        
    def jump(self, A, odd_even_index, index, odd_even, cache):
        if index == len(A)-1:
            return True
        
        if index == None:
            return False
        
        else:
            if odd_even == 1:
                next_index = odd_even_index[index][0]
            else:
                next_index = odd_even_index[index][1]
            
            out = self.jump(A, odd_even_index, next_index, 1-odd_even, cache) if (next_index, 1-odd_even) not in cache else cache[(next_index, 1-odd_even)]
            
            cache[(index, odd_even)] = out
            return out
        
    def oddEvenJumps(self, A):
        odd_even_index = [(-1, -1)]*len(A)
        root = None
        
        for i in reversed(range(len(A))):
            curr_left, curr_right = None, None
            root, curr_left, curr_right = self.insert(root, A[i], i)
            odd_even_index[i] = (curr_left, curr_right)
            
        cache, cnt = {}, 0
        for i in range(len(A)):
            out = self.jump(A, odd_even_index, i, 1, cache)
            if out:
                cnt += 1
        
        return cnt
