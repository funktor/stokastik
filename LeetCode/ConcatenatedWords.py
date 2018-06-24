import collections

class Solution(object):
    def is_concatenated(self, word, original_word, prefixes, word_set):
        if word in word_set and word != original_word:
            return True

        if word[0] not in prefixes:
            return False

        else:
            out = False

            for pos in range(1, len(word)):
                prefix = word[:pos]

                if prefix in prefixes:
                    if prefix in word_set:
                        suffix = word[len(prefix):]
                        out = out or self.is_concatenated(suffix, original_word, prefixes, word_set)

                else:
                    break

            return out


    def findAllConcatenatedWordsInADict(self, words):
        words = [word for word in words if len(word) > 0]
        word_set = set(words)

        prefixes = set()

        for word in words:
            for pos in range(1, len(word) + 1):
                prefix = word[:pos]
                prefixes.add(prefix)

        out = []

        for word in words:
            if self.is_concatenated(word, word, prefixes, word_set):
                out.append(word)

        return out

sol = Solution()
print sol.findAllConcatenatedWordsInADict(["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"])