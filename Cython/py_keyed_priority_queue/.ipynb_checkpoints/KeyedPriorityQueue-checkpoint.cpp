#include "KeyedPriorityQueue.h"

KeyedPriorityQueue::KeyedPriorityQueue () {}

KeyedPriorityQueue::KeyedPriorityQueue (std::vector<HeapNode> heap) {
    for (int i=0; i < static_cast<int>(heap.size()); i++) {
        this->push(heap[i]);
    }
}

KeyedPriorityQueue::~KeyedPriorityQueue () {}
            
int KeyedPriorityQueue::size () {
    return static_cast<int>(this->heap.size());
}

bool KeyedPriorityQueue::comparator(int i, int j) {
    int n = this->size();

    bool cond1 = i < n && j < n && this->heap[i].a > this->heap[j].a;
    bool cond2 = i < n && j < n && this->heap[i].a == this->heap[j].a && this->heap[i].b < this->heap[j].b;

    return cond1 || cond2;
}

int KeyedPriorityQueue::swap (int i, int j) {
    HeapNode x = this->heap[i];
    HeapNode y = this->heap[j];
    
    HeapNode temp1 = x;
    int temp2 = this->heap_ref[x.c];
    
    this->heap[i] = y;
    this->heap_ref[x.c] = this->heap_ref[y.c];
    
    this->heap[j] = temp1;
    this->heap_ref[y.c] = temp2;

    return j;
}

void KeyedPriorityQueue::adjust_parent (int index) {
    while (index > 0 && this->comparator(index/2, index)){
        index = this->swap(index, index/2);
    }
}

void KeyedPriorityQueue::adjust_child (int index) {
    int n = this->size();

    while (true) {
        bool a = 2*index+1 < n && this->comparator(index, 2*index+1);
        bool b = 2*index+2 < n && this->comparator(index, 2*index+2);

        if (a && b) {
            if (this->comparator(2*index+1, 2*index+2)) {
                index = this->swap(index, 2*index+2);
            }
            else {
                index = this->swap(index, 2*index+1);
            }
        }
        else if (a) {
            index = this->swap(index, 2*index+1);
        }
        else if (b) {
            index = this->swap(index, 2*index+2);
        }
        else {
            break;
        }
    }
}

HeapNode KeyedPriorityQueue::remove (int index) {
    HeapNode out = this->heap[index];

    if (index < this->size()-1) {
        this->heap[index] = this->heap.back();
        HeapNode x = this->heap[index];
        this->heap_ref[x.c] = index;
    }

    this->heap.pop_back();
    this->heap_ref.erase(this->heap_ref.find(out.c));

    if (index < this->size()) {
        if (index > 0 && this->comparator(index/2, index)) {
            this->adjust_parent(index);
        }
        else {
            this->adjust_child(index);
        }
    }

    return out;
}

HeapNode KeyedPriorityQueue::pop () {
    return this->remove(0);
}

HeapNode KeyedPriorityQueue::pop (int key) {
    HeapNode out = {0, 0, -1};
    if (this->heap_ref.find(key) != this->heap_ref.end()) {
        return this->remove(this->heap_ref[key]);
    }
    return out;
}

HeapNode KeyedPriorityQueue::get (int key) {
    HeapNode out = {0, 0, -1};
    if (this->heap_ref.find(key) != this->heap_ref.end()) {
        return this->heap[this->heap_ref[key]];
    }

    return out;
}

void KeyedPriorityQueue::push (HeapNode value) {
    int key = value.c;
    if (this->heap_ref.find(key) == this->heap_ref.end()) {
        this->heap.push_back(value);
        this->heap_ref[value.c] = this->size()-1;
        this->adjust_parent(this->size()-1);
    }
}

void KeyedPriorityQueue::set (int key, HeapNode value) {
    if (this->heap_ref.find(key) != this->heap_ref.end()) {
        this->pop(this->heap_ref[key]);
        this->push(value);
    }
}