import collections, heapq

class KeyedPriorityQueue(object):
    def __init__(self, heap):
        self.heap = heap
        self.heap_ref = {(x[0], x[1]):i for i, x in enumerate(heap)}
            
    def getHeapSize(self):
        return len(self.heap)
    
    def swap(self, i, j):
        x, y = self.heap[i], self.heap[j]

        temp1, temp2 = x, self.heap_ref[(x[0], x[1])]
        self.heap[i], self.heap_ref[(x[0], x[1])] = y, self.heap_ref[(y[0], y[1])]
        self.heap[j], self.heap_ref[(y[0], y[1])] = temp1, temp2
        
        return j
    
    def pop(self, index):
        out = self.heap[index]
        self.heap_ref.pop((out[0], out[1]))
        
        if index < len(self.heap)-1:
            self.heap[index] = self.heap[-1]
            x = self.heap[index]
            self.heap_ref[(x[0], x[1])] = index
        
        self.heap.pop()
        
        if index < len(self.heap):
            curr_idx = index

            if curr_idx > 0 and self.heap[curr_idx/2][0] > self.heap[curr_idx][0]:
                while curr_idx > 0 and self.heap[curr_idx/2][0] < self.heap[curr_idx][0]:
                    curr_idx = self.swap(curr_idx, curr_idx/2)

            else:
                while True:
                    a = 2*curr_idx+1 < len(self.heap) and self.heap[curr_idx][0] > self.heap[2*curr_idx+1][0]
                    b = 2*curr_idx+2 < len(self.heap) and self.heap[curr_idx][0] > self.heap[2*curr_idx+2][0]

                    if a or b:
                        if a:
                            curr_idx = self.swap(curr_idx, 2*curr_idx+1)
                        else:
                            curr_idx = self.swap(curr_idx, 2*curr_idx+2)
                    else:
                        break
        
        return out
        
    def getValue(self, key):
        if key in self.heap_ref:
            return self.heap[self.heap_ref[key]]
        
        return None
    
    def setValue(self, key, value):
        if key in self.heap_ref:
            i = self.heap_ref[key]
            self.pop(i)

            self.heap.append(value)
            self.heap_ref[(value[0], value[1])] = len(self.heap)-1

            curr_idx = len(self.heap)-1

            while curr_idx > 0 and self.heap[curr_idx/2][0] > self.heap[curr_idx][0]:
                curr_idx = self.swap(curr_idx, curr_idx/2)
                
    def addValue(self, value):
        self.heap.append(value)
        self.heap_ref[(value[0], value[1])] = len(self.heap)-1

        curr_idx = len(self.heap)-1

        while curr_idx > 0 and self.heap[curr_idx/2][0] > self.heap[curr_idx][0]:
            curr_idx = self.swap(curr_idx, curr_idx/2)
            

class Solution(object):
    def minRefuelStops(self, target, startFuel, stations):
        stations = [(0, startFuel)] + stations
        n = len(stations)
        
        keyedPriorityQueue = KeyedPriorityQueue([(-1, 0, stations[0][1])])
        
        while keyedPriorityQueue.getHeapSize() > 0:
            step, curr_station_id, curr_fuel_capacity = keyedPriorityQueue.pop(0)
            
            if curr_fuel_capacity >= target-stations[curr_station_id][0]:
                return step+1
            
            for j in range(curr_station_id+1, n):
                station_dist, station_fuel = stations[j]
                
                if curr_fuel_capacity >= station_dist-stations[curr_station_id][0]:
                    a = curr_fuel_capacity-station_dist+stations[curr_station_id][0]+station_fuel
                    q = keyedPriorityQueue.getValue((step+1, j))
                    
                    if q is not None:
                        if a > q[2]:
                            keyedPriorityQueue.setValue((step+1, j), (step+1, j, a))
                    else:
                        keyedPriorityQueue.addValue((step+1, j, a))
                else:
                    break
        
        return -1
