class BST(object):
    def __init__(self, value=None, position=None):
        self.value, self.position = value, position
        self.left, self.right = None, None
        self.left_size, self.right_size = 0, 0
        self.parent = None


class Solution(object):
    def create_bst_from_sorted_arr(self, sorted_arr, left, right, node_pos_map):
        if left <= right:
            mid = (left + right) / 2

            node = BST(sorted_arr[mid][1], sorted_arr[mid][0])
            node_pos_map[sorted_arr[mid][0]] = node

            a = self.create_bst_from_sorted_arr(sorted_arr, left, mid - 1, node_pos_map)
            b = self.create_bst_from_sorted_arr(sorted_arr, mid + 1, right, node_pos_map)

            if a is not None:
                a.parent = node

            if b is not None:
                b.parent = node

            node.left, node.right = a, b

            return node
        else:
            return None

    def assign_sizes(self, node):
        if node is not None:
            self.assign_sizes(node.left)
            self.assign_sizes(node.right)

            if node.left is not None:
                node.left_size = node.left.left_size + node.left.right_size
            else:
                node.left_size = 1

            if node.right is not None:
                node.right_size = node.right.left_size + node.right.right_size
            else:
                node.right_size = 1

    def get_common_root(self, root, val1, val2):
        if root is not None:
            if root.value <= val1 and root.value >= val2:
                return root
            else:
                if root.value < val1 and root.value < val2:
                    return self.get_common_root(root.right, val1, val2)
                else:
                    return self.get_common_root(root.left, val1, val2)
        else:
            return root

    def get_position_less(self, root, common_root, offset, upper):
        if root is None:
            return 0
        else:
            if root.value + offset <= upper:
                if root == common_root:
                    return 1 + self.get_position_less(root.right, common_root, offset, upper)
                else:
                    return root.left_size + self.get_position_less(root.right, common_root, offset, upper)

            else:
                return self.get_position_less(root.left, common_root, offset, upper)

    def get_position_more(self, root, common_root, offset, lower):
        if root is None:
            return 0
        else:
            if root.value + offset >= lower:
                if root == common_root:
                    return 1 + self.get_position_more(root.left, common_root, offset, lower)
                else:
                    return root.right_size + self.get_position_more(root.left, common_root, offset, lower)

            else:
                return self.get_position_more(root.right, common_root, offset, lower)

    def adjust_sizes(self, node):
        if node is not None and node.parent is not None:
            if node.parent.left == node:
                node.parent.left_size -= 1
            else:
                node.parent.right_size -= 1

            self.adjust_sizes(node.parent)

    def delete_node(self, root, node, node_pos_map):
        new_node = None

        if node.left is not None and node.right is not None:
            temp = node.right
            node.right_size -= 1

            while temp is not None and temp.left is not None:
                temp.left_size -= 1
                temp = temp.left

            node.value, node.position = temp.value, temp.position
            node_pos_map[temp.position] = node

            node = temp
            new_node = temp.right

        elif node.left is None and node.right is not None:
            new_node = node.right

        elif node.right is None and node.left is not None:
            new_node = node.left

        if node.parent is not None:
            if node.parent.left == node:
                node.parent.left = new_node
            else:
                node.parent.right = new_node

        else:
            root = new_node

        if new_node is not None:
            new_node.parent = node.parent

        return root

    def countRangeSum(self, nums, lower, upper):
        sums = []
        for idx in range(len(nums)):
            if idx == 0:
                sums.append((idx, nums[idx]))
            else:
                sums.append((idx, nums[idx] + sums[idx - 1][1]))

        sums = sorted(sums, key=lambda k: k[1])

        node_pos_map = dict()

        root = self.create_bst_from_sorted_arr(sums, 0, len(sums) - 1, node_pos_map)
        self.assign_sizes(root)

        count, offset = 0, 0

        for idx in range(len(nums)):
            common_root = self.get_common_root(root, upper - offset, lower - offset)
            x, y = self.get_position_less(common_root, common_root, offset, upper), self.get_position_more(common_root,
                                                                                                           common_root,
                                                                                                           offset,
                                                                                                           lower)
            out = max(0, x + y - 1)
            count += out
            self.adjust_sizes(node_pos_map[idx])
            root = self.delete_node(root, node_pos_map[idx], node_pos_map)
            offset -= nums[idx]

        return count

sol = Solution()
print sol.countRangeSum([-2, 5, 1, -1, 0, 2, -3, 2, 4, -2], -5, 5)
