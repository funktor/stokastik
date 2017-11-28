import collections

class Node(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.next = None

class Solution(object):
    def addBoldTag(self, s, dict):
        if len(dict) == 0:
            return ""

        prefix_map = collections.defaultdict(set)

        for word in dict:
            curr = ""
            for pos in range(len(word)):
                curr += word[pos]
                prefix_map[curr].add(word)

        head, tail = None, None
        last_start, pos = 0, 0

        while pos < len(s):
            curr = s[last_start:pos + 1]

            if curr in prefix_map:
                if curr in prefix_map[curr]:
                    node = Node(last_start, pos)

                    if tail is not None:
                        tail.next = node

                    tail = node

                    if head is None:
                        head = node

                pos += 1
            else:
                last_start += 1

                if last_start > pos:
                    pos += 1

        node = head
        while node is not None:
            if node.next is not None and node.next.start <= node.end + 1:
                node.end = node.next.end
                node.next = node.next.next
            else:
                node = node.next

        out_str = ""
        node = head
        last = 0
        while node is not None:
            start, end = node.start, node.end
            out_str += s[last:start] + "<b>" + s[start:end + 1] + "</b>"
            last = end + 1
            node = node.next

        out_str += s[last:len(s)]

        return out_str

sol = Solution()
print sol.addBoldTag("aaabbcc", ["a","b","c"])



