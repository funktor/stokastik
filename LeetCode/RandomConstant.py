import collections, random
from random import randint

class Node(object):
    def __init__(self, val):
        self.val = val
        self.next, self.prev = None, None
        
class RandomizedCollection(object):

    def __init__(self):
        self.tail = None
        self.obj_map = dict()
        self.node_list = []

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
            
        self.obj_map[val].add(node)
        self.node_list.append(node)
        
        return out
        

    def remove(self, val):
        if val in self.obj_map:
            node = self.obj_map[val].pop()
            node.val = self.tail.val
            self.obj_map[node.val].add(node)
            self.obj_map[node.val].remove(self.tail)
            
            self.tail = self.tail.prev
            
            if self.tail is not None:
                self.tail.next = None
            
            if len(self.node_list) > 0:
                self.node_list.pop()
                
            if len(self.obj_map[val]) == 0:
                self.obj_map.pop(val)
            
            return True
        return False
        

    def getRandom(self):
        if len(self.node_list) > 0:
            return random.choice(self.node_list).val
        return None
