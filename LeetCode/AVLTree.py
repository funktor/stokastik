class TreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left, self.right = left, right
        self.height = 0
        
class AVLTree(object):
    def get_tree(self, root):
        if root is None:
            return []
        
        l = self.get_tree(root.left)
        r = self.get_tree(root.right)
        
        return [(root.val, root.height)] + l + r
    
    def get_height(self, node):
        a = node.left.height if node.left is not None else 0
        b = node.right.height if node.right is not None else 0
        
        return 1 + max(a, b)
    
    def insert(self, root, val):
        if root is None:
            root = TreeNode(val)
            root.height = 1
            
        else:
            if val < root.val:
                root.left = self.insert(root.left, val)
            else:
                root.right = self.insert(root.right, val)
            
            a = root.left.height if root.left is not None else 0
            b = root.right.height if root.right is not None else 0
            
            c = root.left.left.height if root.left is not None and root.left.left is not None else 0
            d = root.left.right.height if root.left is not None and root.left.right is not None else 0
            
            e = root.right.left.height if root.right is not None and root.right.left is not None else 0
            f = root.right.right.height if root.right is not None and root.right.right is not None else 0
            
            root.height = 1 + max(a, b)
        
            if b - a == -2:
                if d - c == -1:
                    node = TreeNode(root.val, root.left.right, root.right)
                    node.height = self.get_height(node)
                    root.left.right = node
                    root.left.height = self.get_height(root.left)
                    return root.left
                
                elif d - c == 1:
                    node = TreeNode(root.left.val, root.left.left, root.left.right.left)
                    node.height = self.get_height(node)
                    root.left = root.left.right
                    root.left.left = node
                    
                    node = TreeNode(root.val, root.left.right, root.right)
                    node.height = self.get_height(node)
                    root.left.right = node
                    root.left.height = self.get_height(root.left)
                    return root.left
                
            elif b - a == 2:
                if f - e == 1:
                    node = TreeNode(root.val, root.left, root.right.left)
                    node.height = self.get_height(node)
                    root.right.left = node
                    root.right.height = self.get_height(root.right)
                    return root.right
                
                elif f - e == -1:
                    node = TreeNode(root.right.val, root.right.left.right, root.right.right)
                    node.height = self.get_height(node)
                    root.right = root.right.left
                    root.right.right = node
                    
                    node = TreeNode(root.val, root.left, root.right.left)
                    node.height = self.get_height(node)
                    root.right.left = node
                    root.right.height = self.get_height(root.right)
                    return root.right
                
        return root
