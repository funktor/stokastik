import collections

def bfs(int[:,:] edge_graph, int start, int end, int n):
    queue, visited = collections.deque([start]), set()
    
    if start == end:
        return True
    
    while len(queue) > 0:
        curr = queue.pop()
        visited.add(curr)
        
        for i in range(n):
            if edge_graph[curr][i] == 1 and i not in visited:
                if i == end:
                    return True
                queue.append(i)
                visited.add(i)
    
    return False
        