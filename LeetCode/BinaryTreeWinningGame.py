# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections, numpy as np

class Solution(object):
    def get_neighbors(self, root, parent, neighbors):
        if root is not None:
            neighbors[root.val] = set()
            
            if parent is not None:
                neighbors[root.val].add(parent.val)
            
            if root.left is not None:
                neighbors[root.val].add(root.left.val)
                self.get_neighbors(root.left, root, neighbors)
                
            if root.right is not None:
                neighbors[root.val].add(root.right.val)
                self.get_neighbors(root.right, root, neighbors)
    
    def distances(self, i, n, neighbors):
        queue = collections.deque([(i, 0)])
        visited = set()
        
        dists = [float("Inf")]*(n+1)
        
        while len(queue) > 0:
            ind, dist = queue.popleft()
            dists[ind] = dist
            
            visited.add(ind)
            
            if ind in neighbors:
                for j in neighbors[ind]:
                    if j not in visited:
                        queue.append((j, dist+1))
                        visited.add(j)
        
        return dists
    
    def btreeGameWinningMove(self, root, n, x):
        neighbors = {}
        self.get_neighbors(root, None, neighbors)
        
        distances = [[n+1]*(n+1) for i in range(n+1)]
        
        for i in range(1, n+1):
            queue = collections.deque([(i, 0)])
            visited = set()

            while len(queue) > 0:
                ind, dist = queue.popleft()
                distances[i][ind] = dist

                visited.add(ind)

                if ind in neighbors:
                    for j in neighbors[ind]:
                        if j not in visited:
                            queue.append((j, dist+1))
                            visited.add(j)
        
        
        distances = np.array(distances)
        distances = distances - distances[x]
        
        w = np.sum(distances < 0, axis=1)
        return np.sum(w > n/2) > 0
