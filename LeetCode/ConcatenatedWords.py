import collections

class PrefixTree(object):
    def __init__(self, val):
        self.val = val
        self.children = []

    def add_word_to_tree(self, root, word, node_map):
        for pos in range(1, len(word) + 1):
            prefix = word[:pos]
            if prefix in node_map:
                root = node_map[prefix]

            else:
                node = PrefixTree(word[pos - 1])
                root.children.append(node)
                node_map[prefix] = node
                root = node

        node = PrefixTree('~')
        root.children.append(node)
        node_map[word + '~'] = node


    def compress_tree(self, root):
        queue = collections.deque([root])

        while len(queue) > 0:
            node = queue.popleft()

            if len(node.children) == 1 and node.children[0].val != '~':
                node.val += node.children[0].val

                new_children = []
                for child in node.children:
                    new_children += child.children

                node.children = new_children
                queue.append(node)

            else:
                for child in node.children:
                    queue.append(child)


class Solution(object):
    def is_concatenated(self, word, original_word, prefix_tree, word_set):
        if word in word_set and word != original_word:
            return True

        if word[0] not in prefix_tree:
            return False

        else:
            out, pref = False, word[0]
            prefix = ''

            while pref in prefix_tree:
                prefix += prefix_tree[pref].val
                suffix = word[len(prefix):]

                if len(suffix) == 0:
                    break

                if prefix in word_set and word[:len(prefix)] == prefix:
                    out = out or self.is_concatenated(suffix, original_word, prefix_tree, word_set)

                pref = prefix + suffix[0]

            return out

    def findAllConcatenatedWordsInADict(self, words):
        words = [word for word in words if len(word) > 0]
        word_set = set(words)

        temp = []
        for word in words:
            for pos in range(1, len(word)):
                if word[:pos] in word_set and word[pos:] in word_set:
                    temp.append(word)
                    break

        words_set = word_set.difference(set(temp))
        out = []

        if len(word_set) > 0:
            node_map, prefix_tree = collections.defaultdict(), collections.defaultdict()
            tree = PrefixTree('')

            for word in words_set:
                tree.add_word_to_tree(tree, word, node_map)

            tree.compress_tree(tree)

            for word in words_set:
                if self.is_concatenated(word, word, node_map, word_set):
                    out.append(word)

        return out + temp