# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def list_2_bst(self, head):
        if head is None:
            return None
        
        if head.next is None:
            return TreeNode(head.val)
        
        ptr1, ptr2, prev_ptr1 = head, head, None
        
        while ptr2.next is not None:
            if ptr2.next.next is None:
                ptr2 = ptr2.next
                prev_ptr1 = prev_ptr1.next if prev_ptr1 is not None else head
                ptr1 = ptr1.next
                break
                
            else:
                ptr2 = ptr2.next.next
                prev_ptr1 = prev_ptr1.next if prev_ptr1 is not None else head
                ptr1 = ptr1.next
        
        root = TreeNode(ptr1.val)
        prev_ptr1.next = None
        
        root.left = self.list_2_bst(head)
        root.right = self.list_2_bst(ptr1.next)
        
        return root
        
    def sortedListToBST(self, head):
        return self.list_2_bst(head)
        
