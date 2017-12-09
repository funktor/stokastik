import collections

class Node(object):
    def __init__(self, tag, val):
        self.val = val
        self.tag = tag
        self.children = []

class Solution(object):
    def add_children(self, words, is_root):

        if len(words) == 1 and is_root is False:
            word = words[0]
            length = len(word[0])

            if length <= 2:
                w = word[0]
            else:
                w = str(length - 1) + word[0][length - 1]

            node = Node((w, word[1]), (word[0], word[0][length - 1], length))

            return [node]

        else:
            words_map = collections.defaultdict(int)
            out = []

            for word in words:
                length = len(word[0])
                key = (word[0][0], word[0][length - 1], length)

                if key in words_map:
                    index = words_map[key]
                    out[index][1].append((word[0][1:], word[1]))
                else:
                    out.append((key, [(word[0][1:], word[1])]))
                    words_map[key] = len(out) - 1

            children = []

            for key, d_words in out:
                node = Node("", key)
                node.children = self.add_children(d_words, False)
                children.append(node)

            return children

    def generate_abbv(self, root):
        if root.tag != "":
            return [root.tag]

        out = []
        for child in root.children:
            q = self.generate_abbv(child)

            for x in q:
                out += [(root.val[0] + x[0], x[1])]

        return out

    def wordsAbbreviation(self, words):
        if len(words) == 0:
            return []

        root = Node(None, None)

        children = self.add_children(zip(words, range(len(words))), True)
        root.children = children

        out = []
        for child in root.children:
            out += self.generate_abbv(child)

        final = [""]*len(out)

        for word, pos in out:
            final[pos] = word

        return final


sol = Solution()
print sol.wordsAbbreviation(["like","god","internal","me","internet","interval","intension","face","intrusion"])