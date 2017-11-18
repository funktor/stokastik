class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def construct_tree(self, nums, left, right):
        if left >= right:
            node = TreeNode(nums[left])
        else:
            max_val, max_idx = -float("Inf"), -1

            for idx in range(left, right + 1):
                if nums[idx] > max_val:
                    max_val = nums[idx]
                    max_idx = idx

            node = TreeNode(max_val)

            if left <= max_idx - 1 < len(nums):
                node.left = self.construct_tree(nums, left, max_idx - 1)
            if 0 <= max_idx + 1 <= right:
                node.right = self.construct_tree(nums, max_idx + 1, right)

        return node


    def constructMaximumBinaryTree(self, nums):
        if len(nums) == 1:
            return TreeNode(nums[0])

        tree = self.construct_tree(nums, 0, len(nums)-1)

        return tree


sol = Solution()
out = sol.constructMaximumBinaryTree([3,2,1,6,0,5])


