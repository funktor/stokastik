class Solution(object):
    def findRedundantDirectedConnection(self, edges):
        parents = dict()
        a, b, c, flag = -1, -1, float("Inf"), False
        
        for i in range(len(edges)):
            x, y = edges[i]
            if y not in parents or parents[y][0] == x:
                parents[y] = (x, i)

                w = x
                while w in parents:
                    w = parents[w][0]
                    if w == x or w == y:
                        if a != -1:
                            return edges[a]
                        else:
                            flag = True
                            c = min(c, i)
                            break
                    
            else:
                a, b = parents[y][1], i
        
        if a != -1:
            return edges[a] if flag else edges[b]
        else:
            return edges[c]
