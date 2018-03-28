class Solution(object):
    def is_subsequence(self, s1, s2):
        i, j = 0, 0

        while i < len(s1) and j < len(s2):
            if s1[i] == s2[j]:
                i += 1
                j += 1
            else:
                i += 1

        return j == len(s2)

    def getMaxRepetitions(self, s1, n1, s2, n2):
        m = int((n1 * len(s1)) / (n2 * len(s2))) + 1
        s1_full, s2_full = s1 * n1, s2 * n2

        left, right = 0, m

        while left <= right:
            mid = int((left + right) / 2)

            s2_x = s2_full * mid

            if self.is_subsequence(s1_full, s2_x):
                left = mid + 1
            else:
                right = mid - 1

        return left - 1