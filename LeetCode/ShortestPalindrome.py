class Solution(object):
    def shortestPalindrome(self, s):
        if len(s) == 0:
            return ""

        max_end_palin_idx = -1

        running_left, running_right = '', ''

        for end in reversed(range(len(s))):
            mid = end / 2
            length = end + 1

            if len(running_left) == 0 and len(running_right) == 0:
                if length % 2 == 0:
                    running_left += s[:mid + 1]
                    running_right += s[mid + 1:end + 1][::-1]
                else:
                    running_left += s[:mid]
                    running_right += s[mid + 1:end + 1][::-1]

            if running_left == running_right:
                max_end_palin_idx = end
                break
            else:
                if length % 2 == 0:
                    running_left = running_left[:-1]
                    running_right = running_right[1:]
                else:
                    running_right = running_right[1:] + s[mid]


        non_palin = s[max_end_palin_idx + 1:]
        rev = non_palin[::-1]

        return rev + s