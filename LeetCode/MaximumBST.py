import collections

class Solution(object):
    def is_bst(self, root, cache):
        if root is None:
            cache[root] = (True, 0)
            return True, 0, float("Inf"), -float("Inf")
        else:
            a, x, min_val_l, max_val_l = self.is_bst(root.left, cache)
            b, y, min_val_r, max_val_r = self.is_bst(root.right, cache)

            c = max_val_l < root.val <= min_val_r

            min_val = min(min(root.val, min_val_l), min_val_r)
            max_val = max(max(root.val, max_val_l), max_val_r)

            s = x + y + 1

            if a and b and c:
                cache[root] = (True, s)
                return True, s, min_val, max_val

            cache[root] = (False, s)
            return False, s, min_val, max_val

    def largestBSTSubtree(self, root):
        if root is None:
            return 0

        cache = collections.defaultdict()
        self.is_bst(root, cache)

        queue = [root]
        max_len = 0

        while len(queue) > 0:
            q = queue.pop()

            if cache[q][0]:
                max_len = max(max_len, cache[q][1])

            if q.left is not None:
                queue.insert(0, q.left)

            if q.right is not None:
                queue.insert(0, q.right)

        return max_len