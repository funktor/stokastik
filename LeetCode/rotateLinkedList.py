# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        if head is None:
            return head

        list_len = 0

        node = head
        while node is not None:
            node = node.next
            list_len += 1

        if k % list_len == 0:
            return head

        k = k % list_len

        pt_1, pt_2 = head, head

        dist = 0
        while dist < k:
            pt_2 = pt_2.next
            dist += 1

        while pt_2.next is not None:
            pt_1 = pt_1.next
            pt_2 = pt_2.next

        pt_2.next = head
        head = pt_1.next

        pt_1.next = None

        return head
