import collections
class Solution(object):
    def tree_sums(self, root, counts):
        if root is None:
            return 0
        else:
            sum_l = self.tree_sums(root.left, counts)
            sum_r = self.tree_sums(root.right, counts)

            q = sum_l + sum_r + root.val

            counts[q] += 1

            return q

    def findFrequentTreeSum(self, root):

        counts = collections.defaultdict(int)
        self.tree_sums(root, counts)

        max_count, max_count_keys = -float("Inf"), []

        for key, val in counts.iteritems():
            if val >= max_count:
                max_count = val

        for key, val in counts.iteritems():
            if val == max_count:
                max_count_keys.append(key)

        return max_count_keys

sol = Solution()
print sol.findFrequentTreeSum()



