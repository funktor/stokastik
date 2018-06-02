# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def swap(self, node1, node2):
        if node1 is not None and node2 is not None:
            temp = node1.val
            node1.val = node2.val
            node2.val = temp

    def get_min_max(self, root):
        has_defect, last_defect = 0, (None, None)

        if root.left is not None:
            min_left, max_left, has_defect_left, last_defect_left = self.get_min_max(root.left)
        else:
            min_left, max_left, has_defect_left, last_defect_left = None, None, 0, (None, None)

        if root.right is not None:
            min_right, max_right, has_defect_right, last_defect_right = self.get_min_max(root.right)
        else:
            min_right, max_right, has_defect_right, last_defect_right = None, None, 0, (None, None)

        if max_left is not None and max_left.val > root.val:
            last_defect_left = (root, max_left)
            has_defect_left = 1

            has_defect, last_defect = 1, last_defect_left

        if min_right is not None and min_right.val < root.val:
            last_defect_right = (root, min_right)
            has_defect_right = 1

            has_defect, last_defect = 1, last_defect_right

        if has_defect_left == 1 and has_defect_right == 1:
            self.swap(last_defect_left[1], last_defect_right[1])
            has_defect, last_defect = 0, (None, None)

        elif has_defect_left == 1:
            has_defect, last_defect = 1, last_defect_left

        elif has_defect_right == 1:
            has_defect, last_defect = 1, last_defect_right

        a = root

        if min_left is not None and a.val > min_left.val:
            a = min_left
        if min_right is not None and a.val > min_right.val:
            a = min_right

        b = root

        if max_left is not None and b.val < max_left.val:
            b = max_left
        if max_right is not None and b.val < max_right.val:
            b = max_right

        return a, b, has_defect, last_defect

    def recoverTree(self, root):
        a, b, has_defect, last_defect = self.get_min_max(root)

        if has_defect == 1:
            self.swap(last_defect[0], last_defect[1])