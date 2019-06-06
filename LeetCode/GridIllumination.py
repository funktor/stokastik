class Solution(object):
    def darken(self, a1, a2, b1, b2, x, y):
        if x in a1 and y in a1[x]:
            a1[x].remove(y)
            if len(a1[x]) == 0:
                a1.pop(x)
        
        if y in a2 and x in a2[y]:
            a2[y].remove(x)
            if len(a2[y]) == 0:
                a2.pop(y)
                
        if x-y in b1 and (x,y) in b1[x-y]:
            b1[x-y].remove((x,y))
            if len(b1[x-y]) == 0:
                b1.pop(x-y)
                
        if x+y in b2 and (x,y) in b2[x+y]:
            b2[x+y].remove((x,y))
            if len(b2[x+y]) == 0:
                b2.pop(x+y)
    
    def gridIllumination(self, N, lamps, queries):
        a1, a2 = dict(), dict()
        b1, b2 = dict(), dict()
        
        for x, y in lamps:
            if x not in a1:
                a1[x] = set()
            a1[x].add(y)
            
            if y not in a2:
                a2[y] = set()
            a2[y].add(x)
            
            if x-y not in b1:
                b1[x-y] = set()
            b1[x-y].add((x,y))
            
            if x+y not in b2:
                b2[x+y] = set()
            b2[x+y].add((x,y))
        
        output = []
        for x, y in queries:
            if x in a1 or y in a2 or x-y in b1 or x+y in b2:
                output.append(1)
            else:
                output.append(0)
                
            self.darken(a1, a2, b1, b2, x, y)
            self.darken(a1, a2, b1, b2, x-1, y)
            self.darken(a1, a2, b1, b2, x+1, y)
            self.darken(a1, a2, b1, b2, x, y-1)
            self.darken(a1, a2, b1, b2, x, y+1)
            self.darken(a1, a2, b1, b2, x-1, y-1)
            self.darken(a1, a2, b1, b2, x+1, y+1)
            self.darken(a1, a2, b1, b2, x-1, y+1)
            self.darken(a1, a2, b1, b2, x+1, y-1)
        
        return output
