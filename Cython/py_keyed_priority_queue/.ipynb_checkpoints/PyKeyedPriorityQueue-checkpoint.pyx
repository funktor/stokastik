from libcpp.vector cimport vector
import collections

cdef extern from "KeyedPriorityQueue.h":
    cdef struct HeapNode:
        int a, b, c
    
    cdef cppclass KeyedPriorityQueue:
        KeyedPriorityQueue() except +
        KeyedPriorityQueue(vector[HeapNode]) except +
        int size()
        HeapNode pop()
        HeapNode pop(int key)
        HeapNode get(int key)
        void set(int key, HeapNode value)
        void push(HeapNode value)

cdef class PyKeyedPriorityQueue(object):
    cdef KeyedPriorityQueue kpqueue
    
    def __cinit__(self, arr):
        cdef vector[HeapNode] heap_arr
        cdef HeapNode node
        
        for x, y, z in arr:
            node.a, node.b, node.c = x, y, z
            heap_arr.push_back(node)
            
        self.kpqueue = KeyedPriorityQueue(heap_arr)
            
    def size(self):
        return self.kpqueue.size()
    
    def pop(self, key=None):
        cdef HeapNode out
        if key is None:
            out = self.kpqueue.pop()
        else:
            out = self.kpqueue.pop(key)
        return (out.a, out.b, out.c)
        
    def get(self, key):
        cdef HeapNode out = self.kpqueue.get(key)
        return (out.a, out.b, out.c)
    
    def set(self, key, value):
        cdef HeapNode node
        node.a, node.b, node.c = value
        return self.kpqueue.set(key, node)
                
    def push(self, value):
        cdef HeapNode node
        node.a, node.b, node.c = value
        return self.kpqueue.push(node)