import collections

class KeyedPriorityQueue(object):
    def __init__(self, heap):
        self.heap = heap
        self.heap_ref = {x[2]:i for i, x in enumerate(heap)}
            
    def getHeapSize(self):
        return len(self.heap)
    
    def comparator(self, i, j):
        n = len(self.heap)
        
        cond1 = i < n and j < n and self.heap[i][0] > self.heap[j][0]
        cond2 = i < n and j < n and self.heap[i][0] == self.heap[j][0] and self.heap[i][1] < self.heap[j][1]
        
        return cond1 or cond2
    
    def swap(self, i, j):
        x, y = self.heap[i], self.heap[j]

        temp1, temp2 = x, self.heap_ref[x[2]]
        self.heap[i], self.heap_ref[x[2]] = y, self.heap_ref[y[2]]
        self.heap[j], self.heap_ref[y[2]] = temp1, temp2
        
        return j
    
    def adjust_parent(self, index):
        while index > 0 and self.comparator(index/2, index):
            index = self.swap(index, index/2)
            
    def adjust_child(self, index):
        n = len(self.heap)
        
        while True:
            a = 2*index+1 < n and self.comparator(index, 2*index+1)
            b = 2*index+2 < n and self.comparator(index, 2*index+2)
            
            if a and b:
                if self.comparator(2*index+1, 2*index+2):
                    index = self.swap(index, 2*index+2)
                else:
                    index = self.swap(index, 2*index+1)
            elif a:
                index = self.swap(index, 2*index+1)
            elif b:
                index = self.swap(index, 2*index+2)
            else:
                break
    
    def pop(self, index):
        out = self.heap[index]
        
        if index < len(self.heap)-1:
            self.heap[index] = self.heap[-1]
            x = self.heap[index]
            self.heap_ref[x[2]] = index
        
        self.heap.pop()
        self.heap_ref.pop(out[2])
        
        if index < len(self.heap):
            if index > 0 and self.comparator(index/2, index):
                self.adjust_parent(index)
            else:
                self.adjust_child(index)
                
        return out
        
    def getValue(self, key):
        if key in self.heap_ref:
            return self.heap[self.heap_ref[key]]
        
        return None
    
    def setValue(self, key, value):
        if key in self.heap_ref:
            self.pop(self.heap_ref[key])
            self.addValue(value)
                
    def addValue(self, value):
        self.heap.append(value)
        self.heap_ref[value[2]] = len(self.heap)-1

        self.adjust_parent(len(self.heap)-1)

class Solution(object):
    def minRefuelStops(self, target, startFuel, stations):
        stations = [(0, startFuel)] + stations
        n = len(stations)
        
        keyedPriorityQueue = KeyedPriorityQueue([(-1, startFuel, 0)])
        
        while keyedPriorityQueue.getHeapSize() > 0:
            step, max_dist, curr_station_id = keyedPriorityQueue.pop(0)
            
            if max_dist >= target:
                return step+1

            for j in range(curr_station_id+1, n):
                station_dist, station_fuel = stations[j]

                if max_dist >= station_dist:
                    a, q = max_dist+station_fuel, keyedPriorityQueue.getValue(j)
                    if q is not None:
                        if a > q[1] and q[1] < target:
                            keyedPriorityQueue.setValue(j, (step+1, a, j))
                        else:
                            break
                    else:
                        keyedPriorityQueue.addValue((step+1, a, j))
                else:
                    break
                        
        return -1
