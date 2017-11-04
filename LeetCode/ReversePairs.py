class BST(object):
    def __init__(self):
        self.val, self.position, self.left, self.right, self.size = None, set(), None, None, None


    def print_tree(self, root):
        if root is not None:
            print(root.val, root.position, root.size)

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

        if full_node is None:
            root.size -= 1
        else:
            root.size -= len(full_node.position)

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
            root.val, root.position, root.size = mid_val, position_map[mid_val], len(position_map[mid_val])

            return root

        else:
            mid_idx = len(sorted_nums) / 2
            mid_val = sorted_nums[mid_idx]

            root = BST()
            root.val, root.position, root.size = mid_val, position_map[mid_val], len(position_map[mid_val])

            if mid_idx > 0:
                root.left = self.balance_tree(sorted_nums[:mid_idx], position_map)
                root.size += root.left.size

            if mid_idx + 1 < len(sorted_nums):
                root.right = self.balance_tree(sorted_nums[mid_idx + 1:], position_map)
                root.size += root.right.size

            return root


    def construct_tree(self, nums):
        position_map = dict()

        for idx in range(len(nums)):
            if nums[idx] not in position_map:
                position_map[nums[idx]] = set()

            position_map[nums[idx]].add(idx)

        return self.balance_tree(sorted(set(nums)), position_map)


class Solution(object):

    def search_reverse_pairs_bst(self, num, bst):
        if bst is None:
            return 0

        elif bst.val > num:
            out = len(list(bst.position))

            if bst.right is not None:
                out += bst.right.size

            if bst.left is not None:
                out += self.search_reverse_pairs_bst(num, bst.left)

            return out
        else:
            return self.search_reverse_pairs_bst(num, bst.right)


    def reversePairs_BST(self, nums):
        if len(nums) == 0:
            return 0

        bst = BST()
        root = bst.construct_tree(nums)

        count = 0

        for idx in reversed(range(len(nums))):
            root = bst.delete_node(root, nums[idx], idx)
            count += self.search_reverse_pairs_bst(2 * nums[idx], root)

        return count


    def get_first_valid_position(self, nums, chk_num, left, right):
        if left >= right:
            return left

        mid = int((left + right) / 2)

        if nums[mid] > 2 * chk_num and ((mid > 0 and nums[mid - 1] <= 2 * chk_num) or mid == 0):
            return mid
        elif nums[mid] > 2 * chk_num:
            return self.get_first_valid_position(nums, chk_num, left, mid - 1)
        else:
            return self.get_first_valid_position(nums, chk_num, mid + 1, right)


    def search_reverse_pairs_msort(self, nums):
        if len(nums) == 0:
            return [], 0
        elif len(nums) == 1:
            return [nums[0]], 0

        mid = int(len(nums) / 2)

        a, x = self.search_reverse_pairs_msort(nums[:mid])
        b, y = self.search_reverse_pairs_msort(nums[mid:])

        count = 0
        i, j = self.get_first_valid_position(a, b[0], 0, len(a) - 1), 0

        while i < len(a) and j < len(b):
            if i < len(a) and j < len(b) and 2 * b[j] < a[i]:
                count += len(a) - i
                j += 1
            else:
                i += 1

        out = []

        i, j = 0, 0

        while i < len(a) and j < len(b):
            if i < len(a) and j < len(b) and b[j] < a[i]:
                out.append(b[j])
                j += 1
            else:
                out.append(a[i])
                i += 1

        if i < len(a):
            out += a[i:]
        elif j < len(b):
            out += b[j:]

        return out, count + x + y

    def reversePairs_MSORT(self, nums):
        out, counts = self.search_reverse_pairs_msort(nums)
        return counts

sol = Solution()
print(sol.reversePairs_MSORT([1,3,2,3,1]))