class LRUCache(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.map = dict()
        self.head, self.tail = None, None
        
    def adjust(self, key, value=None):
        out, prev, next = self.map[key]
        self.map.pop(key)
        
        if prev in self.map:
            self.map[prev] = (self.map[prev][0], self.map[prev][1], next)
        else:
            self.head = next

        if next in self.map:
            self.map[next] = (self.map[next][0], prev, self.map[next][2])
        else:
            self.tail = prev
            
        if value is None:
            self.put(key, out)
        else:
            self.put(key, value)
        
        return out

    def get(self, key):
        if key in self.map:
            out = self.adjust(key)
            return out
        return -1
        

    def put(self, key, value):
        if len(self.map) == 0:
            self.head, self.tail = key, key
            self.map[key] = (value, None, None)
            
        elif key not in self.map:
            self.map[key] = (value, self.tail, None)
            self.map[self.tail] = (self.map[self.tail][0], self.map[self.tail][1], key)
            self.tail = key
            
        else:
            self.adjust(key, value)
                
        if len(self.map) > self.capacity:
            next_key = self.map[self.head][2]
            self.map.pop(self.head)
            self.head = next_key
