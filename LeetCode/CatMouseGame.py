import collections, copy, time, numpy as np

class Solution(object):
    def floydWarshall(self, graph):
        n = len(graph)
        dist_mat = np.empty((n, n))
        dist_mat.fill(float("Inf"))

        for i in range(n):
            for j in range(n):
                if j == i:
                    dist_mat[i][i] = 0
                elif j in set(graph[i]):
                    dist_mat[i][j] = 1

        for k in range(n): 
            for i in range(n):
                for j in range(n):
                    dist_mat[i][j] = min(dist_mat[i][j], dist_mat[i][k] + dist_mat[k][j]) 

        return dist_mat
        
    def dfs_search(self, graph, dist_mat, start_node, move, visited, cache):
        if (start_node[0], start_node[1], move) in visited:
            return 0, visited[(start_node[0], start_node[1], move)]
        
        result = set()
        out = None
        
        visited[(start_node[0], start_node[1], move)] = time.time()
        curr_visited = copy.deepcopy(visited)
        
        repeated_pos_t = -float("Inf")
        
        if move == 1:
            if 0 in set(graph[start_node[0]]):
                result.add(1)
            else:
                w = [(end_node, dist_mat[0][end_node]-dist_mat[start_node[1]][end_node]) for end_node in graph[start_node[0]]]
                w = sorted(w, key=lambda k:k[1])
                
                for end_node, _ in w:
                    if end_node == start_node[1]:
                        result.add(2)
                    else:
                        if (end_node, start_node[1], 2) in cache:
                            x = cache[(end_node, start_node[1], 2)]
                        else:
                            x, y = self.dfs_search(graph, dist_mat, (end_node, start_node[1]), 2, curr_visited, cache)
                            repeated_pos_t = max(repeated_pos_t, y)
                            
                        result.add(x)
                        if x == 1:
                            break
            
            if 1 in result:
                out = 1
            elif 0 in result:
                out = 0
            else:
                out = 2
            
        else:
            w = set([end_node for end_node in graph[start_node[1]]])
            
            if start_node[0] in w:
                result.add(2)
            else:
                v = [(end_node, dist_mat[start_node[0]][end_node]) for end_node in graph[start_node[1]]]
                v = sorted(v, key=lambda k:k[1])
                
                for end_node, _ in v:
                    if end_node != 0:
                        if (start_node[0], end_node, 1) in cache:
                            x = cache[(start_node[0], end_node, 1)]
                        else:
                            x, y = self.dfs_search(graph, dist_mat, (start_node[0], end_node), 1, curr_visited, cache)
                            repeated_pos_t = max(repeated_pos_t, y)

                        result.add(x)
                        if x == 2:
                            break
                            
            if 2 in result:
                out = 2
            elif 0 in result:
                out = 0
            else:
                out = 1
        
        if out != 0:
            cache[(start_node[0], start_node[1], move)] = out
        else:
            if visited[(start_node[0], start_node[1], move)] <= repeated_pos_t:
                cache[(start_node[0], start_node[1], move)] = 0
        
        return out, repeated_pos_t
        
    def catMouseGame(self, graph):
        visited, cache = dict(), dict()
        dist_mat = self.floydWarshall(graph)
        
        out, _ = self.dfs_search(graph, dist_mat, (1,2), 1, visited, cache)
        return out