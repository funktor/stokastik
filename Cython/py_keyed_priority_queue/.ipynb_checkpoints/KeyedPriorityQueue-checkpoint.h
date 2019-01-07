#ifndef KEYEDPRIORITYQUEUE_H
#define KEYEDPRIORITYQUEUE_H

#include <iostream> 
#include <map>
#include <vector>

struct HeapNode {
    int a;
    int b;
    int c;
};

class KeyedPriorityQueue {
    private:
        std::vector<HeapNode> heap;
        std::map<int, int> heap_ref;
        bool comparator(int i, int j);
        int swap(int i, int j);
        void adjust_parent(int index);
        void adjust_child(int index);
        HeapNode remove(int index);
    
    public:
        KeyedPriorityQueue();
        KeyedPriorityQueue(std::vector<HeapNode> heap);
        ~KeyedPriorityQueue();
        int size();
        HeapNode pop();
        HeapNode pop(int key);
        void push(HeapNode value);
        HeapNode get(int key);
        void set(int key, HeapNode value);
        
};

#endif