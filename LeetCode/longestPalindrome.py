class Solution(object):

    def is_palindrome(self, s):
        return (len(s) % 2 == 0 and s[:len(s)/2] == s[len(s)/2:][::-1]) \
               or (len(s) % 2 == 1 and s[:len(s)/2] == s[len(s)/2 + 1:][::-1])

    def get_palindrome(self, s, length):
        for idx in range(len(s) - length + 1):
            w = s[idx:idx + length]

            if self.is_palindrome(w):
                return w

        return -1

    def longestPalindrome(self, s):
        if len(s) == 0:
            return ""

        left, right = 1, len(s)

        while left <= right:
            mid = (left + right) / 2
            out = self.get_palindrome(s, mid)

            if out != -1:
                left = mid + 1
            else:
                if mid + 1 <= len(s) and self.get_palindrome(s, mid + 1) != -1:
                    left = mid + 2
                else:
                    right = mid - 1

        if left > 1:
            return self.get_palindrome(s, left - 1)
        else:
            return self.get_palindrome(s, left)

sol = Solution()
print sol.longestPalindrome("cbbd")
