import numpy as np

class Node(object):
    def __init__(self, val):
        self.val = val
        self.prev, self.next = None, None
        
def get_linked_list(n):
    arr = np.empty(n+1, dtype=np.int32)
    arr.fill(1)
    arr[0], arr[1] = 0, 0
    
    node_ref, curr_node = np.empty(len(arr), dtype=Node), None
    
    for i in range(2, len(arr)):
        if i == 2:
            node = Node((arr[i], i))
        else:
            node = Node((arr[i], i))
            curr_node.next = node
            node.prev = curr_node
            
        curr_node = node
        node_ref[i] = curr_node
    
    return node_ref

def sieve(n):
    node_ref = get_linked_list(n)
    sqrt_n, curr_node = int(np.sqrt(n)), node_ref[2]
    
    while curr_node is not None and curr_node.val[1] <= sqrt_n:
        i = curr_node.val[1]
        j = i**2
        while j <= n:
            if node_ref[j] is not None:
                del_node = node_ref[j]

                if del_node.prev is not None:
                    del_node.prev.next = del_node.next
                if del_node.next is not None:
                    del_node.next.prev = del_node.prev
                node_ref[j] = None
            j += i
            
        curr_node = curr_node.next

    return np.nonzero(node_ref)