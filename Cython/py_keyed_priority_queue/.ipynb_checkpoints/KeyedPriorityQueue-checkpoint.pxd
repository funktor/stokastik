from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "KeyedPriorityQueue.h" namespace "kp_queue":
    cdef struct HeapNode:
        pass
    
    cdef cppclass KeyedPriorityQueue:
        KeyedPriorityQueue() except +
        KeyedPriorityQueue(vector[HeapNode]) except +
        int getHeapSize()
        bool comparator(int i, int j)
        int swap(int i, int j)
        void adjust_parent(int index)
        void adjust_child(int index)
        HeapNode pop(int index)
        HeapNode getValue(int key)
        void setValue(int key, HeapNode value)
        void addValue(HeapNode value)