class Solution(object):
    def can_extend(self, char_q, new_char):
        a = set(char_q)
        return len(a) < 2 or (len(a) == 2 and new_char in a)

    def lengthOfLongestSubstringTwoDistinct(self, s):
        if len(s) <= 2:
            return len(s)

        start, end = 0, 1

        char_q = [s[start]]
        char_q.insert(0, s[end])

        curr_max_len = 2

        while start <= end and end < len(s):
            if (end + 1 < len(s) and self.can_extend(char_q, s[end + 1])) or end == start:
                curr_max_len = max(curr_max_len, end - start + 2)
                end += 1

                if end < len(s):
                    char_q.insert(0, s[end])
            else:
                if len(char_q) > 0:
                    char_q.pop()
                start += 1

        return curr_max_len