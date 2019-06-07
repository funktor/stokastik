import collections, numpy as np
class Solution(object):
    def bfs(self, B, start, visited):
        queue = collections.deque([start])
        visited.add(start)
        
        while len(queue) > 0:
            q = queue.popleft()
            j = [i for i in range(B.shape[0]) if i not in visited]
            out = np.nonzero(np.sum(B[j]!=B[q], axis=1) <= 2)[0]
            for i in out:
                queue.append(j[i])
                visited.add(j[i])
        
        
    def numSimilarGroups(self, A):
        B = []
        for x in A:
            t = []
            for y in x:
                t.append(ord(y)-ord('a'))
            B.append(t)
        
        B = np.array(B)
        
        visited = set()
        num_grps = 0
        for i in range(len(A)):
            if i not in visited:
                num_grps += 1
                self.bfs(B, i, visited)
        
        return num_grps
