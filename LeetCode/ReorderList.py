import collections


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def re_order(self, head, last_pos, positions):
        if head is not None and head.next is not None:
            second_last = positions[last_pos - 1]
            flag = False if last_pos == int(len(positions) / 2) else True

            if flag:
                next_head = head.next

                head.next = second_last.next
                second_last.next = None

                if head.next is not None:
                    head.next.next = self.re_order(next_head, last_pos - 1, positions)

        return head

    def reorderList(self, head):
        if head is not None:
            positions = []

            temp = head
            while temp is not None:
                positions.append(temp)
                temp = temp.next

            m = int(len(positions) / 2)
            m = m + 1 if len(positions) % 2 == 1 else m

            cached = collections.defaultdict(ListNode)

            for pos in reversed(range(m)):
                if pos == m - 1:
                    cached[pos] = positions[pos]

                    if len(positions) % 2 == 1:
                        cached[pos].next = None
                    else:
                        cached[pos].next.next = None

                else:
                    a, b = positions[pos], positions[len(positions) - pos - 1]
                    a.next = b
                    a.next.next = cached[pos + 1]

                    cached[pos] = a

            head = cached[0]
