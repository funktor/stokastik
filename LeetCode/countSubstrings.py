import collections

class Solution(object):
    def countSubstrings(self, s):
        counts = 0
        is_palin = collections.defaultdict(dict)

        for length in range(1, len(s) + 1):
            for start in range(len(s) - length + 1):
                end = start + length - 1

                if length == 1:
                    is_palin[start][end] = True
                elif length == 2:
                    is_palin[start][end] = s[start] == s[end]
                else:
                    is_palin[start][end] = is_palin[start + 1][end - 1] and s[start] == s[end]

                if is_palin[start][end]:
                    counts += 1

        return counts

sol = Solution()
print sol.countSubstrings("abc")

