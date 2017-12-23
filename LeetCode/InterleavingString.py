import collections

class Solution(object):
    def interleave(self, s1, s2, s3, pos1, pos2, pos3, cache):
        if pos1 >= len(s1) and pos2 >= len(s2):
            return pos3 >= len(s3)
        elif pos1 >= len(s1):
            return "".join(s2[pos2:]) == "".join(s3[pos3:])
        elif pos2 >= len(s2):
            return "".join(s1[pos1:]) == "".join(s3[pos3:])
        else:
            if s1[pos1] == s2[pos2] == s3[pos3]:
                if pos1 + 1 in cache and pos2 in cache[pos1 + 1]:
                    a = cache[pos1 + 1][pos2]
                else:
                    a = self.interleave(s1, s2, s3, pos1 + 1, pos2, pos3 + 1, cache)
                    cache[pos1 + 1][pos2] = a

                if pos1 in cache and pos2 + 1 in cache[pos1]:
                    b = cache[pos1][pos2 + 1]
                else:
                    b = self.interleave(s1, s2, s3, pos1, pos2 + 1, pos3 + 1, cache)
                    cache[pos1][pos2 + 1] = b

                return a or b

            elif s1[pos1] == s3[pos3]:
                if pos1 + 1 in cache and pos2 in cache[pos1 + 1]:
                    return cache[pos1 + 1][pos2]
                else:
                    cache[pos1 + 1][pos2] = self.interleave(s1, s2, s3, pos1 + 1, pos2, pos3 + 1, cache)
                    return cache[pos1 + 1][pos2]

            elif s2[pos2] == s3[pos3]:
                if pos1 in cache and pos2 + 1 in cache[pos1 + 1]:
                    return cache[pos1][pos2 + 1]
                else:
                    cache[pos1][pos2 + 1] = self.interleave(s1, s2, s3, pos1, pos2 + 1, pos3 + 1, cache)
                    return cache[pos1][pos2 + 1]

            else:
                return False


    def isInterleave(self, s1, s2, s3):
        cache = collections.defaultdict(dict)

        return self.interleave(s1, s2, s3, 0, 0, 0, cache)


sol = Solution()
print sol.isInterleave("abbbbbbcabbacaacccababaabcccabcacbcaabbbacccaaaaaababbbacbb",
"ccaacabbacaccacababbbbabbcacccacccccaabaababacbbacabbbbabc",
"cacbabbacbbbabcbaacbbaccacaacaacccabababbbababcccbabcabbaccabcccacccaabbcbcaccccaaaaabaaaaababbbbacbbabacbbacabbbbabc")