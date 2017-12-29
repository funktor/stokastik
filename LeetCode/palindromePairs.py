import collections

class Solution(object):
    def palindromePairs(self, words):
        palin_pre_map, palin_pos_map = collections.defaultdict(set), collections.defaultdict(set)

        out = set()

        palin, blank_pos = [], -1

        for idx in range(len(words)):
            word = words[idx]
            if len(word) == 0:
                blank_pos = idx
            elif word[::-1] == word:
                palin.append(idx)

        if blank_pos != -1:
            for x in palin:
                out.add((blank_pos, x))
                out.add((x, blank_pos))

        for idx in range(len(words)):
            word = words[idx]

            for pos in range(len(word)):
                a, b = word[:pos], word[pos:]

                if a == a[::-1]:
                    palin_pre_map[b].add(idx)

                if b == b[::-1]:
                    palin_pos_map[a].add(idx)

        for idx in range(len(words)):
            word = words[idx]
            word_rev = word[::-1]

            if word_rev in palin_pre_map:
                for x in palin_pre_map[word_rev]:
                    if x != idx:
                        out.add((idx, x))

            if word_rev in palin_pos_map:
                for x in palin_pos_map[word_rev]:
                    if x != idx:
                        out.add((x, idx))

        final_out = []

        for x in out:
            final_out.append([x[0], x[1]])

        return final_out