import collections

class Solution(object):
    def get_word_start_pos(self, s, words):

        wordDict, prefix_map = set(words), collections.defaultdict(set)

        for word in wordDict:
            for pos in range(1, len(word) + 1):
                a = word[:pos]
                prefix_map[a].add(word)


        pos_word_map = collections.defaultdict()

        for idx in range(len(s)):
            if s[idx] in prefix_map:
                pos_word_map[idx] = s[idx]

        curr_length = 1
        flag = True

        while flag:
            temp_map = collections.defaultdict()
            flag = False

            for index, word in pos_word_map.items():
                if index + curr_length < len(s):
                    new_word = word + s[index + curr_length]
                    if new_word in prefix_map:
                        flag = True
                        temp_map[index] = new_word
                    else:
                        if word in wordDict:
                            temp_map[index] = word
                else:
                    if word in wordDict:
                        temp_map[index] = word

            if flag:
                pos_word_map = temp_map
                curr_length += 1

        return pos_word_map


    def findSubstring(self, s, words):
        if len(s) == 0 or len(words) == 0:
            return []

        words_count = collections.defaultdict(int)

        for word in words:
            words_count[word] += 1

        word_lengths = [len(word) for word in words]

        if max(word_lengths) > len(s):
            return []

        pos_word_map = self.get_word_start_pos(s, words)

        pos_arr = sorted([pos for pos, word in pos_word_map.items()])

        word_len, start_indices = len(words[0]), []

        for i in range(pos_arr[0], pos_arr[0] + word_len):
            start = i

            queue = []
            running_cnts = collections.defaultdict(int)

            while start < len(s):
                if start in pos_word_map:
                    wd = pos_word_map[start]

                    if wd in running_cnts and running_cnts[wd] >= words_count[wd]:

                        if len(queue) == len(words):
                            start_indices.append(queue[len(queue) - 1])

                        q = queue.pop()

                        if pos_word_map[q] in running_cnts:
                            running_cnts[pos_word_map[q]] -= 1

                    else:
                        queue.insert(0, start)

                        running_cnts[wd] += 1
                        start += word_len
                else:
                    start += word_len

                    if len(queue) == len(words):
                        start_indices.append(queue[len(queue) - 1])

                    queue = []
                    running_cnts = collections.defaultdict(int)

            if len(queue) == len(words):
                start_indices.append(queue[len(queue) - 1])

        return start_indices

sol = Solution()
print sol.findSubstring("mississippi", ["mississippis"])