# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def get_max_sum(self, root):
        if root.left is None and root.right is None:
            return (root.val, root.val)

        elif root.left is None:
            x = self.get_max_sum(root.right)
            w1 = max(root.val, x[0] + root.val)
            w2 = max(x[0], x[1])
            return (w1, w2)

        elif root.right is None:
            x = self.get_max_sum(root.left)
            w1 = max(root.val, x[0] + root.val)
            w2 = max(x[0], x[1])
            return (w1, w2)

        else:
            x = self.get_max_sum(root.left)
            y = self.get_max_sum(root.right)

            a1 = root.val
            a2 = x[0] + root.val
            a3 = y[0] + root.val
            a4 = x[0] + y[0] + root.val
            a5 = x[1]
            a6 = y[1]
            a7 = x[0]
            a8 = y[0]

            b1 = [a1, a2, a3]
            b2 = [a4, a5, a6, a7, a8]

            max_num1, max_num2 = -float("Inf"), -float("Inf")

            for w in b1:
                max_num1 = max(max_num1, w)

            for w in b2:
                max_num2 = max(max_num2, w)

            return (max_num1, max_num2)

    def maxPathSum(self, root):
        if root is not None:
            out = self.get_max_sum(root)
            return max(out[0], out[1])

        return 0