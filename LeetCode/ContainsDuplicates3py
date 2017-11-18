class BST(object):
    def __init__(self):
        self.val, self.position, self.left, self.right = None, set(), None, None

    def add(self, root, node):
        if root is None:
            root = node
        else:
            if root.val == node.val:
                root.position.update(node.position)

            elif root.val > node.val:
                root.left = self.add(root.left, node)

            else:
                root.right = self.add(root.right, node)

        return root


    def print_tree(self, root):
        if root is not None:
            print(root.val, root.position)

            if root.left is None:
                self.print_tree(root.right)
            elif root.right is None:
                self.print_tree(root.left)
            else:
                self.print_tree(root.left)
                self.print_tree(root.right)


    def get_minimum_node(self, root):
        if root.left is None and root.right is None:
            return root
        else:
            if root.left is None:
                return root
            else:
                return self.get_minimum_node(root.left)


    def delete_node(self, root, node_val, node_position, full_node=None):

        if root.val == node_val:
            if node_position != -1:
                root.position.remove(node_position)

            if len(root.position) == 0 or node_position == -1:
                if root.left is None and root.right is None:
                    return None
                elif root.left is None:
                    return root.right
                elif root.right is None:
                    return root.left
                else:
                    min_node_right = self.get_minimum_node(root.right)

                    root.val = min_node_right.val
                    root.position = min_node_right.position

                    root.right = self.delete_node(root.right, min_node_right.val, -1, min_node_right)
            else:
                return root

        else:
            if root.val < node_val and root.right is not None:
                root.right = self.delete_node(root.right, node_val, node_position, full_node)

            elif root.val > node_val and root.left is not None:
                root.left = self.delete_node(root.left, node_val, node_position, full_node)

        return root


    def balance_tree(self, sorted_nums, position_map):

        if len(sorted_nums) == 1:
            mid_val = sorted_nums[0]

            root = BST()
            root.val, root.position = mid_val, position_map[mid_val]

            return root

        else:
            mid_idx = len(sorted_nums) / 2
            mid_val = sorted_nums[mid_idx]

            root = BST()
            root.val, root.position = mid_val, position_map[mid_val]

            if mid_idx > 0:
                root.left = self.balance_tree(sorted_nums[:mid_idx], position_map)

            if mid_idx + 1 < len(sorted_nums):
                root.right = self.balance_tree(sorted_nums[mid_idx + 1:], position_map)

            return root


    def construct_tree(self, nums):
        position_map = dict()

        for idx in range(len(nums)):
            if nums[idx] not in position_map:
                position_map[nums[idx]] = set()

            position_map[nums[idx]].add(idx)

        return self.balance_tree(sorted(set(nums)), position_map)


class Solution(object):

    def found(self, root, t, val, position):
        if root is None:
            return False
        elif abs(root.val - val) <= t and (position not in root.position or len(root.position) > 1):
            return True
        else:
            if root.val > val:
                return self.found(root.left, t, val, position)
            else:
                return self.found(root.right, t, val, position)


    def containsNearbyAlmostDuplicate(self, nums, k, t):

        if len(nums) < 2:
            return False

        if k == 0:
            return True

        bst = BST()
        curr = nums[:k+1]

        tree = bst.construct_tree(curr)

        if self.found(tree, t, nums[0], 0):
            return True

        for idx in range(1, len(nums)):

            if idx + k < len(nums):
                node = BST()
                node.val = nums[idx + k]
                node.position = set([idx + k])

                tree = bst.add(tree, node)

            if idx - k >= 1:
                tree = bst.delete_node(tree, nums[idx - k - 1], idx - k - 1)

            if self.found(tree, t, nums[idx], idx):
                return True

        return False


sol = Solution()
print sol.containsNearbyAlmostDuplicate([1,3,1], 2, 1)