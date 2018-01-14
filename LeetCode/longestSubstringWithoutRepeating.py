import collections

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        if len(s) == 0:
            return 0

        start, end, curr_longest = 0, 1, -float("Inf")
        pos = collections.defaultdict(int)

        pos[s[start]] = start

        while end < len(s):
            if s[end] in pos and pos[s[end]] >= start:
                curr_longest = max(curr_longest, end - start)
                dup_pos = pos[s[end]]
                start = dup_pos + 1

            pos[s[end]] = end
            end += 1

        curr_longest = max(curr_longest, end - start)

        return curr_longest

sol = Solution()
print sol.lengthOfLongestSubstring("a")