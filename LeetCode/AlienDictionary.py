import collections

class Node(object):
    def __init__(self, val):
        self.val = val
        self.parent = set()
        self.children = set()

class Solution(object):

    def get_order(self, node, visited):

        if node.val in visited:
            return None
        elif len(node.children) == 0:
            return node.val
        else:
            out = node.val
            visited.add(out)

            for child in node.children:
                w = self.get_order(child, visited)

                if w is None:
                    return None

                out += ''.join([x for x in w])

            return out


    def alienOrder(self, words):
        if len(words) == 0:
            return ""

        node_map = collections.defaultdict(Node)

        for idx in range(len(words)):

            if idx == len(words) - 1:
                if len(words[idx]) > 0 and words[idx][0] not in node_map:
                    node = Node(words[idx][0])
                    node_map[words[idx][0]] = node
            else:
                word_1, word_2 = words[idx], words[idx + 1]
                min_len = min(len(word_1), len(word_2))

                for pos in range(min_len):
                    if word_1[pos] != word_2[pos]:

                        if word_1[pos] in node_map:
                            node_1 = node_map[word_1[pos]]
                        else:
                            node_1 = Node(word_1[pos])

                        if word_2[pos] in node_map:
                            node_2 = node_map[word_2[pos]]
                        else:
                            node_2 = Node(word_2[pos])

                        node_1.children.add(node_2)
                        node_2.parent.add(node_1)

                        node_map[word_1[pos]] = node_1
                        node_map[word_2[pos]] = node_2

                        break

        root_nodes = set([ch for ch, node in node_map.iteritems() if len(node.parent) == 0])

        all_chars = set()

        for word in words:
            all_chars.update(list(word))

        out = ""

        if len(root_nodes) > 0:
            for root_node in root_nodes:
                q = self.get_order(node_map[root_node], set())
                if q is None:
                    return ""
                out += q

            if len(out) < len(node_map):
                return ""
        else:
            return out

        new_out = ""
        suffix_set = set()

        for ch in reversed(list(out)):
            if ch not in suffix_set:
                new_out += ch
            suffix_set.add(ch)

        out = ''.join([x for x in reversed(list(new_out))])

        q = set(out)

        unknown = ""
        for ch in all_chars:
            if ch not in q:
                unknown += ch

        out += unknown

        return out

sol = Solution()
print sol.alienOrder(["sobdtfqmkx","touaona","adt","sjlz","pofhlg","jwi","g","hnhe","acrciuu","axhchsi","axz"])
