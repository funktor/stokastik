import collections

class Solution(object):
    def get_depth(self, i, edge_dict, depth_dict, visited):
        visited.add(i)
        max_d = 0
        for x in edge_dict[i]:
            if x not in visited:
                d = self.get_depth(x, edge_dict, depth_dict, visited)
                max_d = max(max_d, d)
        
        depth_dict[i] = max_d+1
        return max_d+1
        
    def findMinHeightTrees(self, n, edges):
        if n == 0:
            return []
        
        if len(edges) == 0:
            return range(n)
        
        edge_dict, depth_dict = {}, {}
        
        for x, y in edges:
            if x not in edge_dict:
                edge_dict[x] = []
            edge_dict[x].append(y)
            
            if y not in edge_dict:
                edge_dict[y] = []
            edge_dict[y].append(x)
            
        self.get_depth(0, edge_dict, depth_dict, set())
        
        curr_root, min_depth = 0, depth_dict[0]
        results, visited = {}, set()
        curr_depth = depth_dict[0]
        
        while True:
            if curr_depth not in results:
                results[curr_depth] = []
            results[curr_depth].append(curr_root)
            
            min_depth = min(min_depth, curr_depth)
                
            visited.add(curr_root)
            depths = []
            
            for x in edge_dict[curr_root]:
                depths.append((x, depth_dict[x]))
            
            depths = sorted(depths, key=lambda k:-k[1])
            
            if depths[0][0] not in visited:
                if len(depths) > 1:
                    curr_depth = max(depths[0][1], depths[1][1]+2)
                    depth_dict[curr_root] = depths[1][1]+1
                else:
                    curr_depth = max(depths[0][1], 2)
                    depth_dict[curr_root] = 1
                
                curr_root = depths[0][0]
            else:
                break
        
        return results[min_depth]
