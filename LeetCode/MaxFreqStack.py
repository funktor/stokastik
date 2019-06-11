import heapq
class FreqStack(object):

    def __init__(self):
        self.map1, self.map2 = dict(), dict()
        self.cnt_heap = []

    def push(self, x):
        if x not in self.map1:
            self.map1[x] = 0
        self.map1[x] += 1
        
        if self.map1[x] not in self.map2:
            self.map2[self.map1[x]] = []
            heapq.heappush(self.cnt_heap, -self.map1[x])
            
        self.map2[self.map1[x]].append(x)
        

    def pop(self):
        cnt = -self.cnt_heap[0]
        out = self.map2[cnt].pop()
        
        self.map1[out] -= 1
        if self.map1[out] == 0:
            self.map1.pop(out)
        
        if len(self.map2[cnt]) == 0:
            self.map2.pop(cnt)
            heapq.heappop(self.cnt_heap)
            
        return out
