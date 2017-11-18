class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        alphabets = list(map(chr, range(97, 123)))
        wordList_set = set(wordList)

        queue = [(beginWord, 1)]
        visited = set([beginWord])

        while len(queue) > 0:
            front = queue.pop()
            length = front[1]
            a = front[0]

            if a == endWord:
                return length

            for pos in range(len(a)):
                for alpha in alphabets:
                    new_word = a[:pos] + alpha + a[pos + 1:]
                    if new_word not in visited and new_word in wordList_set:
                        visited.add(new_word)
                        queue.insert(0, (new_word, length + 1))

        return 0

sol = Solution()
print sol.ladderLength("hit", "cog", ["hot","dot","dog","lot","log","cog"])