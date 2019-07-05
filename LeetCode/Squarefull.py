import math, collections, heapq

class Solution(object):
    def bfs(self, A, a_dict):
        heap = [(0, -1, [])]
        visited = set()
        
        cnt = set()
        while len(heap) > 0:
            length, a, curr_seq = heapq.heappop(heap)
            q = [A[i] for i in curr_seq]
            
            if len(curr_seq) == len(A):
                cnt.add(tuple([A[i] for i in curr_seq]))
            
            elif a == -1:
                for x in a_dict.keys():
                    heapq.heappush(heap, (-1, x, [x]))
                    visited.add(tuple([A[x]]))
                    
            else:
                if a in a_dict:
                    for x in a_dict[a]:
                        if x not in curr_seq:
                            new_q = q + [A[x]]
                            if tuple(new_q) not in visited:
                                heapq.heappush(heap, (-(length+1), x, curr_seq + [x]))
                                visited.add(tuple(q))
        
        return len(cnt)
    
    def numSquarefulPerms(self, A):
        a_dict = {}
        for i in range(len(A)):
            if i not in a_dict:
                a_dict[i] = []
            for j in range(len(A)):
                if i != j:
                    s = A[i] + A[j]
                    if int(math.sqrt(s))**2 == s:
                        a_dict[i].append(j)
        
        
        return self.bfs(A, a_dict)
