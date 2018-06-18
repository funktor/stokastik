import collections

class Solution(object):
    def is_concatenated(self, word_idx, words, word_map):
        word = words[word_idx]
        queue = collections.deque([0])

        while len(queue) > 0:
            start = queue.popleft()
            possible_indices = word_map[word_idx][start]

            for end in possible_indices:
                pos_word = word[start : end + 1]

                if pos_word != word:
                    if end == len(word) - 1:
                        return True

                    if end + 1 in word_map[word_idx]:
                        queue.append(end + 1)

        return False


    def findAllConcatenatedWordsInADict(self, words):
        words = [word for word in words if len(word) > 0]
        word_set = set(words)

        word_map = collections.defaultdict(dict)
        valid_soln = []

        for idx in range(len(words)):
            word = words[idx]

            flag = False
            for i in range(len(word)):
                word_map[idx][i] = []

                for j in range(i, len(word)):
                    if word[i : j + 1] in word_set and word[j + 1 :] in word_set and i == 0 and j != len(word) - 1:
                        valid_soln.append(idx)
                        flag = True
                        break

                    elif word[i : j + 1] in word_set:
                        word_map[idx][i].append(j)

                if flag:
                    break

        q = set(valid_soln)

        out = [words[idx] for idx in valid_soln]

        for word_idx in range(len(words)):
            if word_idx not in q and self.is_concatenated(word_idx, words, word_map):
                out.append(words[word_idx])

        return out