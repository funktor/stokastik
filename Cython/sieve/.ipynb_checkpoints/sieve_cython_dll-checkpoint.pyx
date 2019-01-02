from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

cdef struct Node:
    int index
    Node *prev
    Node *next
    
ctypedef Node* myNode

def sieve(int n):
    cdef int sqrt_n, i, j
    cdef myNode curr_node, del_node
    
    cdef myNode *node_ref = <myNode*> malloc((n+1) * sizeof(myNode))
    
    node_ref[0], node_ref[1] = NULL, NULL
    
    for i in range(2, n+1):
        node_ref[i] = <myNode> malloc(sizeof(Node))
        node_ref[i].index = i
        node_ref[i].prev, node_ref[i].next = NULL, NULL
        
        if i > 2:
            curr_node.next = node_ref[i]
            node_ref[i].prev = curr_node
            
        curr_node = node_ref[i]
    
    sqrt_n, curr_node = int(sqrt(n)), node_ref[2]
    
    while curr_node is not NULL and curr_node.index <= sqrt_n:
        i = curr_node.index
        j = i**2
        while j <= n:
            if node_ref[j] is not NULL:
                del_node = node_ref[j]

                if del_node.prev is not NULL:
                    del_node.prev.next = del_node.next
                if del_node.next is not NULL:
                    del_node.next.prev = del_node.prev
                node_ref[j] = NULL
            j += i
            
        curr_node = curr_node.next

    cdef list out = [i for i in range(2, n+1) if node_ref[i] is not NULL]
    free(node_ref)
    return out