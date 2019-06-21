class Solution(object):
    def min_camera(self, root):
        if root is None:
            return 0, 0, float("Inf")
        
        al, bl, cl = self.min_camera(root.left)
        ar, br, cr = self.min_camera(root.right)
        
        a = bl + br
        b = min(cl+br, cl+cr, cr+bl)
        c = 1 + min(al, bl, cl) + min(ar, br, cr)
        
        return a, b, c
        
    def minCameraCover(self, root):
        a, b, c = self.min_camera(root)
        return min(b, c)
