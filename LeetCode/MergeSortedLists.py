class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        if len(lists) == 0:
            return []

        root, new_node = None, None
        curr_nodes = []

        for lst in lists:
            curr_nodes.append(lst)

        indices = set(range(len(curr_nodes)))

        while True:
            min_list_idx = -1
            min_list_val = float("inf")

            for idx in indices:
                lst = curr_nodes[idx]

                if lst is not None and lst.val <= min_list_val:
                    min_list_val = lst.val
                    min_list_idx = idx

            if min_list_idx != -1:
                if root is not None:
                    new_node.next = ListNode(min_list_val)
                    new_node = new_node.next
                else:
                    root = ListNode(min_list_val)
                    new_node = root

                curr_nodes[min_list_idx] = curr_nodes[min_list_idx].next

                if curr_nodes[min_list_idx] is None:
                    indices.remove(min_list_idx)
            else:
                break

        return root

lsts = []

lst_nodes = []

for lst in lsts:
    root, node = None, None

    for x in lst:
        if root is None:
            root = ListNode(x)
            node = root
        else:
            node.next = ListNode(x)
            node = node.next

    if root is not None:
        lst_nodes.append(root)

sol = Solution()
out = sol.mergeKLists(lst_nodes)

node = out
while node is not None:
    print(node.val)
    node = node.next


