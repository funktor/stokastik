import collections, random
from random import randint

class Node(object):
    def __init__(self, val):
        self.val = val
        self.next, self.prev = None, None
        
class RandomizedCollection(object):

    def __init__(self):
        self.head, self.tail = None, None
        self.obj_map, self.idx_map, self.idx_map2 = dict(), dict(), dict()
        self.n = 0

    def insert(self, val):
        node = Node(val)
        
        if self.tail is None:
            self.tail = node
            
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = self.tail.next
        
        out = val not in self.obj_map
        if out:
            self.obj_map[val] = set()
            
        self.obj_map[val].add(self.n)
        
        self.idx_map[self.n] = node
        self.idx_map2[node] = self.n
        
        self.n += 1
        
        return out
        

    def remove(self, val):
        if val in self.obj_map:
            idx = self.obj_map[val].pop()
            
            curr_node = self.idx_map[idx]
            curr_node.val = self.tail.val
            
            self.tail = self.tail.prev
            
            if len(self.obj_map[val]) == 0:
                self.obj_map.pop(val)
            
            if curr_node.val in self.obj_map and self.n-1 in self.obj_map[curr_node.val]:
                self.obj_map[curr_node.val].remove(self.n-1)
                
            if idx < self.n-1:
                if curr_node.val not in self.obj_map:
                    self.obj_map[curr_node.val] = set()
                    
                self.obj_map[curr_node.val].add(idx)
                
            self.n -= 1
            
            return True
        return False
        

    def getRandom(self):
        if self.n > 0:
            return self.idx_map[randint(0, self.n-1)].val
        return None
